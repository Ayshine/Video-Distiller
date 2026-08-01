"""Shared test fixtures.

Generates a small synthetic video with ffmpeg so tests never need a real video.
The clip is a test pattern with a silent audio track — enough to exercise
ingest (audio extraction + probing) and to stand in as an input path for
transcribe (whose ASR is mocked).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_VIDEO = FIXTURE_DIR / "sample.mp4"
FIXTURE_DURATION_S = 5  # short: keeps ingest tests fast

CARDS_VIDEO = FIXTURE_DIR / "cards.mp4"
# Three visually distinct static patterns, then a repeat of the first — so
# scene detection sees 4 scenes but perceptual-hash dedup collapses to 3.
CARD_PATTERNS = ["smptebars", "rgbtestsrc", "yuvtestsrc", "smptebars"]
CARD_DURATION_S = 3


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _generate_fixture(path: Path) -> None:
    """Create a test-pattern video with a silent audio track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={FIXTURE_DURATION_S}:size=320x240:rate=15",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={FIXTURE_DURATION_S}",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate fixture video:\n{result.stderr}")


@pytest.fixture(scope="session")
def sample_video() -> Path:
    """Path to a synthetic sample video, generated once per session."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not available; skipping tests that need a fixture video")
    if not FIXTURE_VIDEO.exists():
        _generate_fixture(FIXTURE_VIDEO)
    return FIXTURE_VIDEO


def _generate_cards(path: Path) -> None:
    """Build one clip of distinct static cards (last repeats the first).

    Uses a single re-encoding ffmpeg command with the concat *filter* (not the
    demuxer + stream-copy, which can emit broken PTS that OpenCV rejects).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    for pattern in CARD_PATTERNS:
        cmd += [
            "-f",
            "lavfi",
            "-i",
            f"{pattern}=duration={CARD_DURATION_S}:size=320x240:rate=15",
        ]
    streams = "".join(f"[{i}:v]" for i in range(len(CARD_PATTERNS)))
    cmd += [
        "-filter_complex",
        f"{streams}concat=n={len(CARD_PATTERNS)}:v=1:a=0[out]",
        "-map",
        "[out]",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate cards video:\n{result.stderr}")


@pytest.fixture(scope="session")
def cards_video() -> Path:
    """A video of 3 distinct cards + 1 duplicate, for scene-detection tests."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not available; skipping tests that need a fixture video")
    if not CARDS_VIDEO.exists():
        _generate_cards(CARDS_VIDEO)
    return CARDS_VIDEO
