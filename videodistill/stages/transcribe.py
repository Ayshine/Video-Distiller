"""Stage 2 — transcribe.

Two ASR backends, chosen by the configured model name:

- **local** faster-whisper (``tiny``/``base``/``small``/…): free, offline, but
  weaker on non-English audio.
- **hosted** OpenAI (``whisper-1``): far more accurate, keeps timestamps, and
  takes the profile-vocabulary prompt — reached through the provider abstraction
  so no SDK is called from this stage.

Either way the output is timestamped :class:`~videodistill.models.TranscriptSegment`
records. The ASR prompt comes from the domain profile's vocabulary (none when
empty), so no domain terms appear in this code.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, cast

from videodistill.errors import ProviderError, VideoDistillError
from videodistill.language import normalize_language
from videodistill.llm.base import LLMClient
from videodistill.models import Transcript, TranscriptSegment, VideoMeta, Word
from videodistill.profile import DomainProfile

IS_STUB = False

logger = logging.getLogger("videodistill.stages.transcribe")

_DEFAULT_MODEL_SIZE = "small"

# Model names served by the OpenAI provider (rather than local faster-whisper).
OPENAI_ASR_MODELS = {"whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"}

# The OpenAI transcription API rejects uploads over 25 MB. A 16 kHz mono WAV is
# ~1.9 MB/min, so we split long audio into ~10-minute chunks (~19 MB) and stitch
# the results back with each chunk's time offset applied.
OPENAI_MAX_UPLOAD_BYTES = 24 * 1024 * 1024
CHUNK_SECONDS = 600


class _WhisperLike(Protocol):
    """Structural type for the subset of WhisperModel we use."""

    def transcribe(self, audio: str, **kwargs: Any) -> tuple[Any, Any]: ...


def run(
    video_meta: VideoMeta,
    job_dir: Path,
    profile: DomainProfile,
    *,
    model_size: str | None = None,
    llm: LLMClient | None = None,
) -> Transcript:
    """Transcribe the audio referenced by ``video_meta`` into ``job_dir``."""
    size = model_size or _DEFAULT_MODEL_SIZE
    prompt = profile.asr_initial_prompt()
    language = (
        normalize_language(profile.asr_language) if profile.asr_language else None
    )

    if size in OPENAI_ASR_MODELS:
        transcript = _transcribe_openai(video_meta, size, prompt, llm, language)
    else:
        transcript = _transcribe_local(video_meta, size, prompt, language)

    transcript.save(job_dir)
    return transcript


def _transcribe_openai(
    video_meta: VideoMeta,
    model: str,
    prompt: str | None,
    llm: LLMClient | None,
    language: str | None,
) -> Transcript:
    if llm is None:
        raise ProviderError(
            f"ASR model '{model}' needs the OpenAI provider (set OPENAI_API_KEY)."
        )
    audio = Path(video_meta.audio_path)

    if audio.stat().st_size <= OPENAI_MAX_UPLOAD_BYTES:
        result = llm.transcribe(audio, model=model, prompt=prompt, language=language)
        detected = result.language
        segments = [
            TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
            for s in result.segments
        ]
    else:
        detected, segments = _transcribe_in_chunks(
            audio, video_meta.duration_s, model, prompt, llm, language
        )

    # A forced language is authoritative; otherwise trust what the model detected.
    return Transcript(
        language=language or normalize_language(detected),
        model_size=model,
        segments=segments,
    )


def _transcribe_in_chunks(
    audio: Path,
    duration_s: float,
    model: str,
    prompt: str | None,
    llm: LLMClient,
    language: str | None,
) -> tuple[str, list[TranscriptSegment]]:
    """Split audio into <=25 MB chunks, transcribe each, offset timestamps.

    Passing ``language`` to every chunk is what fixes the old bug where a quiet
    first chunk auto-detected the wrong language and its transcript came back
    near-empty (and that guess then labelled the whole transcript).
    """
    detected = language or "unknown"
    segments: list[TranscriptSegment] = []
    with tempfile.TemporaryDirectory() as tmp:
        chunks = list(_split_audio(audio, duration_s, Path(tmp)))
        logger.info("transcribe: audio > 25 MB, split into %d chunk(s)", len(chunks))
        for offset, chunk_path in chunks:
            result = llm.transcribe(
                chunk_path, model=model, prompt=prompt, language=language
            )
            if detected == "unknown":
                detected = result.language
            for s in result.segments:
                segments.append(
                    TranscriptSegment(
                        start=s.start + offset,
                        end=s.end + offset,
                        text=s.text.strip(),
                    )
                )
    return detected, segments


def _split_audio(
    audio: Path, duration_s: float, tmp_dir: Path
) -> Iterator[tuple[float, Path]]:
    """Yield (offset_seconds, chunk_path) for fixed-length slices of the audio."""
    offset = 0.0
    index = 0
    while offset < duration_s:
        chunk = tmp_dir / f"chunk_{index:03d}.wav"
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{offset:.3f}",
                "-t",
                str(CHUNK_SECONDS),
                "-i",
                str(audio),
                "-c",
                "copy",
                str(chunk),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not chunk.exists() or chunk.stat().st_size < 1024:
            if index == 0:
                raise VideoDistillError(
                    f"ffmpeg failed to split audio:\n{result.stderr.strip()}"
                )
            break
        yield offset, chunk
        offset += CHUNK_SECONDS
        index += 1


def _transcribe_local(
    video_meta: VideoMeta, size: str, prompt: str | None, language: str | None
) -> Transcript:
    model = _build_model(size)
    segments_iter, info = model.transcribe(
        video_meta.audio_path,
        word_timestamps=True,
        initial_prompt=prompt,
        vad_filter=True,
        language=language,  # None lets faster-whisper auto-detect
    )
    segments = [_to_segment(seg) for seg in segments_iter]
    return Transcript(
        language=language or normalize_language(getattr(info, "language", "unknown")),
        model_size=size,
        segments=segments,
    )


def _build_model(model_size: str) -> _WhisperLike:
    """Instantiate a faster-whisper model. Patched out in tests."""
    from faster_whisper import WhisperModel

    # int8 on CPU is the portable default; callers can override via env later.
    return cast(
        _WhisperLike, WhisperModel(model_size, device="cpu", compute_type="int8")
    )


def _to_segment(seg: Any) -> TranscriptSegment:
    """Convert a faster-whisper segment (duck-typed) into our model."""
    words = [
        Word(
            start=float(w.start),
            end=float(w.end),
            text=w.word,
            probability=getattr(w, "probability", None),
        )
        for w in (getattr(seg, "words", None) or [])
    ]
    return TranscriptSegment(
        start=float(seg.start),
        end=float(seg.end),
        text=seg.text.strip(),
        words=words,
    )
