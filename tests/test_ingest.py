"""Tests for the ingest stage."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from videodistill.errors import VideoDistillError
from videodistill.models import VideoMeta
from videodistill.profile import DomainProfile
from videodistill.stages import ingest

# ingest is domain-independent; any profile will do.
PROFILE = DomainProfile(name="generic")


def test_ingest_produces_meta_and_audio(sample_video: Path, tmp_path: Path) -> None:
    meta = ingest.run(sample_video, tmp_path, PROFILE)

    assert isinstance(meta, VideoMeta)
    assert meta.width == 320
    assert meta.height == 240
    assert meta.fps == pytest.approx(15, abs=0.5)
    assert meta.duration_s > 0

    audio_path = Path(meta.audio_path)
    assert audio_path.exists()
    with wave.open(str(audio_path), "rb") as wav:
        assert wav.getframerate() == 16_000
        assert wav.getnchannels() == 1


def test_ingest_writes_loadable_artifact(sample_video: Path, tmp_path: Path) -> None:
    ingest.run(sample_video, tmp_path, PROFILE)

    loaded = VideoMeta.load(tmp_path)
    assert loaded.source_path == str(sample_video.resolve())
    assert (tmp_path / "video_meta.json").exists()


def test_ingest_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(VideoDistillError, match="not found"):
        ingest.run(tmp_path / "does_not_exist.mp4", tmp_path, PROFILE)


def test_ingest_reports_missing_ffmpeg(
    sample_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest.shutil, "which", lambda _name: None)
    with pytest.raises(VideoDistillError, match="ffmpeg"):
        ingest.run(sample_video, tmp_path, PROFILE)
