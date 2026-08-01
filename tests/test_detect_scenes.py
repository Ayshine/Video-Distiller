"""Tests for the detect_scenes stage.

The pure helpers (sampling policy, phash dedup) are unit-tested without a video;
the end-to-end path is exercised on the synthetic 3-distinct-cards + 1-duplicate
fixture, where dedup must collapse the duplicate.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import imagehash

from videodistill.models import VideoMeta
from videodistill.profile import DomainProfile
from videodistill.stages import detect_scenes

PROFILE = DomainProfile(name="generic")


# --- pure helpers -----------------------------------------------------------


def test_candidate_timestamps_short_scene_is_just_middle() -> None:
    assert detect_scenes._candidate_timestamps([(0.0, 10.0)]) == [5.0]


def test_candidate_timestamps_long_scene_is_sampled() -> None:
    # A 60s scene (> 45s threshold): middle (30) + every 20s (20, 40).
    assert detect_scenes._candidate_timestamps([(0.0, 60.0)]) == [20.0, 30.0, 40.0]


def test_candidate_timestamps_multiple_scenes() -> None:
    assert detect_scenes._candidate_timestamps([(0.0, 4.0), (4.0, 10.0)]) == [2.0, 7.0]


def test_dedup_drops_near_duplicate_keeps_distinct() -> None:
    h1 = imagehash.hex_to_hash("0000000000000000")
    h2 = imagehash.hex_to_hash("ffffffffffffffff")  # 64 bits from h1
    h3 = imagehash.hex_to_hash("00000000ffffffff")  # 32 bits from h1
    h1_dup = imagehash.hex_to_hash("0000000000000007")  # 3 bits from h1 (<= 6)

    kept = detect_scenes._dedup([h1, h2, h3, h1_dup])

    assert kept == [0, 1, 2]  # the near-duplicate of h1 is dropped


def test_dedup_threshold_is_inclusive_boundary() -> None:
    base = imagehash.hex_to_hash("0000000000000000")
    six_bits = imagehash.hex_to_hash("000000000000003f")  # exactly 6 bits set
    assert (base - six_bits) == 6
    # <= threshold => duplicate (dropped); > threshold => kept.
    assert detect_scenes._dedup([base, six_bits], threshold=6) == [0]
    assert detect_scenes._dedup([base, six_bits], threshold=5) == [0, 1]


# --- end-to-end on the fixture ---------------------------------------------


def test_detect_scenes_collapses_duplicate_card(
    cards_video: Path, tmp_path: Path
) -> None:
    meta = VideoMeta(
        source_path=str(cards_video),
        audio_path=str(tmp_path / "unused.wav"),
        duration_s=12.0,
        width=320,
        height=240,
        fps=15.0,
    )

    result = detect_scenes.run(meta, tmp_path, PROFILE)

    # 4 scenes detected, but the repeated card is deduped away -> 3.
    assert len(result.keyframes) == 3

    # Every kept keyframe wrote a real JPEG.
    for kf in result.keyframes:
        assert Path(kf.image_path).exists()
        assert Path(kf.image_path).suffix == ".jpg"

    # Kept frames are mutually distinct (all pairwise Hamming > threshold).
    hashes = [imagehash.hex_to_hash(kf.phash) for kf in result.keyframes]
    for a, b in itertools.combinations(hashes, 2):
        assert (a - b) > detect_scenes.PHASH_HAMMING_THRESHOLD

    # The artifact round-trips through its canonical path.
    from videodistill.models import KeyframeSet

    assert len(KeyframeSet.load(tmp_path).keyframes) == 3
