"""Tests for the transcribe stage.

faster-whisper is mocked via ``_build_model`` so the test needs neither a model
download nor real audio — it verifies our segment/word conversion, the
profile-driven initial prompt, and artifact writing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from videodistill.errors import ProviderError
from videodistill.llm.base import ASRResult, ASRSegment
from videodistill.models import Transcript, VideoMeta
from videodistill.profile import DomainProfile
from videodistill.stages import transcribe


@dataclass
class _FakeWord:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class _FakeSegment:
    start: float
    end: float
    text: str
    words: list[_FakeWord]


@dataclass
class _FakeInfo:
    language: str


class _FakeWhisper:
    """Stands in for faster_whisper.WhisperModel; records call kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio: str, **kwargs: object):  # noqa: ANN003
        self.calls.append({"audio": audio, **kwargs})
        segments = [
            _FakeSegment(
                start=0.0,
                end=1.5,
                text="  Dereference the pointer.  ",
                words=[
                    _FakeWord(0.0, 0.6, "Dereference", 0.98),
                    _FakeWord(0.6, 1.0, "the", 0.99),
                    _FakeWord(1.0, 1.5, "pointer.", 0.97),
                ],
            ),
            _FakeSegment(start=1.5, end=3.0, text="Use std::vector.", words=[]),
        ]
        return iter(segments), _FakeInfo(language="en")


@pytest.fixture
def video_meta(tmp_path: Path) -> VideoMeta:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake wav bytes")
    return VideoMeta(
        source_path=str(tmp_path / "video.mp4"),
        audio_path=str(audio),
        duration_s=3.0,
        width=320,
        height=240,
        fps=15.0,
    )


def test_transcribe_builds_segments_with_words(
    video_meta: VideoMeta, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeWhisper()
    monkeypatch.setattr(transcribe, "_build_model", lambda _size: fake)

    result = transcribe.run(
        video_meta, tmp_path, DomainProfile(name="generic"), model_size="small"
    )

    assert isinstance(result, Transcript)
    assert result.language == "en"
    assert result.model_size == "small"
    assert len(result.segments) == 2

    first = result.segments[0]
    assert first.text == "Dereference the pointer."  # stripped
    assert len(first.words) == 3
    assert first.words[0].text == "Dereference"
    assert first.words[0].probability == pytest.approx(0.98)
    assert result.segments[1].words == []


def test_transcribe_seeds_prompt_from_profile_vocabulary(
    video_meta: VideoMeta, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeWhisper()
    monkeypatch.setattr(transcribe, "_build_model", lambda _size: fake)

    profile = DomainProfile(
        name="sample", description="a domain", vocabulary=["widget", "foo::bar"]
    )
    transcribe.run(video_meta, tmp_path, profile)

    call = fake.calls[0]
    assert call["word_timestamps"] is True
    assert "foo::bar" in str(call["initial_prompt"])  # domain vocabulary seed


def test_transcribe_empty_vocabulary_passes_no_prompt(
    video_meta: VideoMeta, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeWhisper()
    monkeypatch.setattr(transcribe, "_build_model", lambda _size: fake)

    transcribe.run(video_meta, tmp_path, DomainProfile(name="generic"))

    assert fake.calls[0]["initial_prompt"] is None


def test_transcribe_writes_loadable_artifact(
    video_meta: VideoMeta, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transcribe, "_build_model", lambda _size: _FakeWhisper())

    transcribe.run(video_meta, tmp_path, DomainProfile(name="generic"))

    loaded = Transcript.load(tmp_path)
    assert len(loaded.segments) == 2
    assert (tmp_path / "transcript.json").exists()


class _FakeASR:
    """Stands in for the OpenAI provider's transcribe()."""

    def __init__(self) -> None:
        self.prompt: str | None = None
        self.model: str | None = None
        self.language: str | None = None

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        prompt: str | None = None,
        language: str | None = None,
    ) -> ASRResult:
        self.prompt = prompt
        self.model = model
        self.language = language
        return ASRResult(
            language="tr",
            segments=[
                ASRSegment(0.0, 2.0, "  Merhaba pointer  ", []),
                ASRSegment(2.0, 4.0, "std::vector kullanalim", []),
            ],
        )


def test_hosted_model_routes_through_provider(
    video_meta: VideoMeta, tmp_path: Path
) -> None:
    fake = _FakeASR()
    profile = DomainProfile(name="x", vocabulary=["std::vector", "pointer"])

    result = transcribe.run(
        video_meta, tmp_path, profile, model_size="whisper-1", llm=fake
    )

    assert result.language == "tr"
    assert result.model_size == "whisper-1"
    assert result.segments[0].text == "Merhaba pointer"  # stripped
    assert fake.model == "whisper-1"
    assert "std::vector" in (fake.prompt or "")  # profile vocab became the prompt


def test_hosted_model_without_provider_raises(
    video_meta: VideoMeta, tmp_path: Path
) -> None:
    with pytest.raises(ProviderError):
        transcribe.run(
            video_meta, tmp_path, DomainProfile(name="x"), model_size="whisper-1"
        )


class _ChunkASR:
    """Returns one segment per call so we can check offsets are applied."""

    def __init__(self) -> None:
        self.calls = 0
        self.languages: list[str | None] = []

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        prompt: str | None = None,
        language: str | None = None,
    ) -> ASRResult:
        self.calls += 1
        self.languages.append(language)
        return ASRResult(
            language="tr",
            segments=[ASRSegment(1.0, 2.0, f"chunk{self.calls}", [])],
        )


def test_long_audio_is_chunked_with_offsets(
    video_meta: VideoMeta, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the chunking branch and stub the ffmpeg split with two fake chunks.
    monkeypatch.setattr(transcribe, "OPENAI_MAX_UPLOAD_BYTES", 0)
    c0, c1 = tmp_path / "c0.wav", tmp_path / "c1.wav"
    c0.write_bytes(b"x" * 2048)
    c1.write_bytes(b"y" * 2048)

    def fake_split(audio: Path, duration: float, tmp: Path):  # noqa: ANN202
        yield 0.0, c0
        yield 600.0, c1

    monkeypatch.setattr(transcribe, "_split_audio", fake_split)

    fake = _ChunkASR()
    result = transcribe.run(
        video_meta, tmp_path, DomainProfile(name="x"), model_size="whisper-1", llm=fake
    )

    assert fake.calls == 2  # both chunks transcribed
    assert [round(s.start, 1) for s in result.segments] == [1.0, 601.0]  # 2nd offset
    assert [s.text for s in result.segments] == ["chunk1", "chunk2"]


def test_hosted_model_forwards_forced_language(
    video_meta: VideoMeta, tmp_path: Path
) -> None:
    fake = _FakeASR()
    profile = DomainProfile(name="x", asr_language="tr")
    result = transcribe.run(
        video_meta, tmp_path, profile, model_size="whisper-1", llm=fake
    )
    assert fake.language == "tr"  # profile language reached the provider
    assert result.language == "tr"


def test_forced_language_reaches_every_chunk(
    video_meta: VideoMeta, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The bug this guards: a quiet first chunk auto-detecting the wrong language
    # and dropping content. Forcing it must apply to ALL chunks, not just the 1st.
    monkeypatch.setattr(transcribe, "OPENAI_MAX_UPLOAD_BYTES", 0)
    c0, c1 = tmp_path / "c0.wav", tmp_path / "c1.wav"
    c0.write_bytes(b"x" * 2048)
    c1.write_bytes(b"y" * 2048)
    monkeypatch.setattr(
        transcribe, "_split_audio", lambda a, d, t: iter([(0.0, c0), (600.0, c1)])
    )

    fake = _ChunkASR()
    transcribe.run(
        video_meta,
        tmp_path,
        DomainProfile(name="x", asr_language="tr"),
        model_size="whisper-1",
        llm=fake,
    )
    assert fake.languages == ["tr", "tr"]
