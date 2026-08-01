"""Stage 1 — ingest.

Validate the input video, extract a 16kHz mono WAV suitable for ASR, and probe
duration / resolution / fps. Writes :class:`~videodistill.models.VideoMeta`.

Uses ffmpeg/ffprobe via subprocess (no Python binding), so the only runtime
dependency is the ffmpeg binary being on PATH. The profile is accepted for a
uniform stage signature but ingest is domain-independent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from videodistill.errors import DependencyMissing, VideoDistillError
from videodistill.models import VideoMeta
from videodistill.profile import DomainProfile

IS_STUB = False

_AUDIO_SAMPLE_RATE = 16_000
_AUDIO_CHANNELS = 1


def run(video_path: Path, job_dir: Path, profile: DomainProfile) -> VideoMeta:
    """Extract audio and metadata from ``video_path`` into ``job_dir``."""
    _require_binaries()

    video_path = video_path.expanduser().resolve()
    if not video_path.exists():
        raise VideoDistillError(f"Input video not found: {video_path}")

    job_dir.mkdir(parents=True, exist_ok=True)
    audio_path = job_dir / "audio.wav"

    probe = _probe(video_path)
    _extract_audio(video_path, audio_path)

    meta = VideoMeta(
        source_path=str(video_path),
        audio_path=str(audio_path),
        duration_s=probe["duration_s"],
        width=probe["width"],
        height=probe["height"],
        fps=probe["fps"],
        audio_sample_rate=_AUDIO_SAMPLE_RATE,
        audio_channels=_AUDIO_CHANNELS,
    )
    meta.save(job_dir)
    return meta


def _require_binaries() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise DependencyMissing(
                f"'{binary}' was not found on PATH. Install ffmpeg "
                "(macOS: `brew install ffmpeg`) and try again."
            )


def _probe(video_path: Path) -> dict[str, float | int]:
    """Return duration, resolution and fps using ffprobe's JSON output."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoDistillError(
            f"ffprobe failed for {video_path}:\n{result.stderr.strip()}"
        )

    data: dict[str, Any] = json.loads(result.stdout)
    video_stream = _first_video_stream(data.get("streams", []))
    if video_stream is None:
        raise VideoDistillError(f"No video stream found in {video_path}")

    duration_s = float(data.get("format", {}).get("duration", 0.0))
    if duration_s <= 0:
        # Some containers only carry duration on the stream.
        duration_s = float(video_stream.get("duration", 0.0))

    return {
        "duration_s": duration_s,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": _parse_fps(video_stream.get("avg_frame_rate", "0/0")),
    }


def _first_video_stream(streams: list[dict[str, Any]]) -> dict[str, Any] | None:
    for stream in streams:
        if stream.get("codec_type") == "video":
            return stream
    return None


def _parse_fps(rate: str) -> float:
    """Parse ffprobe's ``num/den`` frame-rate string into a float."""
    try:
        num, den = rate.split("/")
        denominator = float(den)
        if denominator == 0:
            return 0.0
        return float(num) / denominator
    except (ValueError, ZeroDivisionError):
        return 0.0


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract 16kHz mono PCM WAV — the format faster-whisper expects."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(_AUDIO_SAMPLE_RATE),
        "-ac",
        str(_AUDIO_CHANNELS),
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoDistillError(
            f"ffmpeg audio extraction failed for {video_path}:\n{result.stderr.strip()}"
        )
    if not audio_path.exists():
        raise VideoDistillError(
            f"ffmpeg reported success but {audio_path} was not created."
        )
