"""Stage 3 — detect_scenes.

Find the distinct "cards" shown in a video and keep one representative frame
each. Steps:

1. PySceneDetect's ContentDetector finds scene boundaries.
2. One candidate frame is sampled from the middle of each scene; long scenes
   (a presenter scrolling code, say) are sampled every 20s so a slow reveal is
   not collapsed to a single frame.
3. A perceptual-hash pass drops candidates within Hamming distance 6 of a
   frame already kept — this kills cursor/laser-pointer jitter on an otherwise
   static slide, and collapses a scroll that stops moving.

Writes :class:`~videodistill.models.KeyframeSet` (JPEGs live in the job dir) and
logs the raw→unique reduction. Domain-independent; the profile is accepted for
a uniform stage signature.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw

from videodistill.errors import DependencyMissing, VideoDistillError
from videodistill.models import Keyframe, KeyframeSet, VideoMeta
from videodistill.profile import DomainProfile

IS_STUB = False

logger = logging.getLogger("videodistill.stages.detect_scenes")

# A high-resolution (1024-bit) perceptual hash so small-but-important on-screen
# changes register — e.g. a few lines of code typed into an IDE, which a coarse
# 64-bit hash cannot see because the static window chrome dominates it.
PHASH_SIZE = 32
# Hamming distance below which two frames are the same card. Tuned for the
# 1024-bit hash: idle jitter / cursor blink stays under it; a real content
# change (typing, a new slide) is far above.
PHASH_HAMMING_THRESHOLD = 24
# Scenes longer than this are treated as possible scrolls and sampled densely.
SCROLL_SCENE_THRESHOLD_S = 45.0
SCROLL_SAMPLE_INTERVAL_S = 20.0


def run(video_meta: VideoMeta, job_dir: Path, profile: DomainProfile) -> KeyframeSet:
    """Detect scenes in the source video and write deduplicated keyframes."""
    import shutil

    if shutil.which("ffmpeg") is None:
        raise DependencyMissing(
            "'ffmpeg' was not found on PATH. Install ffmpeg "
            "(macOS: `brew install ffmpeg`) and try again."
        )

    video_path = Path(video_meta.source_path)
    if not video_path.exists():
        raise VideoDistillError(f"Source video not found: {video_path}")

    scenes = _detect_scenes(video_path)
    if not scenes:
        # No cuts detected: treat the whole video as one scene.
        scenes = [(0.0, video_meta.duration_s)]

    timestamps = _candidate_timestamps(scenes)

    frames_dir = job_dir / "keyframes"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Extract every candidate frame, then dedup by perceptual hash. Masked
    # regions (e.g. a webcam overlay) come from the profile.
    candidates: list[tuple[float, Path, imagehash.ImageHash]] = []
    for ts in timestamps:
        tmp = frames_dir / f"_cand_{ts:09.3f}.jpg"
        try:
            _extract_frame(video_path, ts, tmp)
        except VideoDistillError as exc:
            # One un-extractable candidate (e.g. a scene cut within a frame of
            # EOF) must not abort the whole stage — skip it and carry on.
            logger.warning("detect_scenes: skipping frame at %.3fs: %s", ts, exc)
            continue
        candidates.append((ts, tmp, _phash(tmp, profile.keyframe_mask_regions)))

    kept_indices = set(_dedup([ph for _, _, ph in candidates]))

    keyframes: list[Keyframe] = []
    for idx, (ts, tmp, ph) in enumerate(candidates):
        if idx in kept_indices:
            final = frames_dir / f"keyframe_{len(keyframes):03d}.jpg"
            tmp.replace(final)
            keyframes.append(
                Keyframe(timestamp=ts, image_path=str(final), phash=str(ph))
            )
        else:
            tmp.unlink(missing_ok=True)

    logger.info(
        "detect_scenes: %d scene(s), %d raw candidate(s) -> %d unique keyframe(s)",
        len(scenes),
        len(candidates),
        len(keyframes),
    )

    result = KeyframeSet(keyframes=keyframes)
    result.save(job_dir)
    return result


def _detect_scenes(video_path: Path) -> list[tuple[float, float]]:
    """Return (start_s, end_s) for each detected scene. Mocked in tests."""
    from scenedetect import ContentDetector, detect

    scene_list = detect(str(video_path), ContentDetector())
    return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]


def _candidate_timestamps(
    scenes: list[tuple[float, float]],
    *,
    scroll_threshold_s: float = SCROLL_SCENE_THRESHOLD_S,
    interval_s: float = SCROLL_SAMPLE_INTERVAL_S,
) -> list[float]:
    """Middle of each scene, plus dense sampling inside long (scroll) scenes.

    Pure function so the sampling policy is unit-testable without a video.
    """
    out: set[float] = set()
    for start, end in scenes:
        duration = end - start
        out.add(round(start + duration / 2, 3))
        if duration > scroll_threshold_s:
            t = start + interval_s
            while t < end:
                out.add(round(t, 3))
                t += interval_s
    return sorted(out)


def _dedup(
    hashes: list[imagehash.ImageHash],
    *,
    threshold: int = PHASH_HAMMING_THRESHOLD,
) -> list[int]:
    """Greedy in-order dedup; returns the indices to keep.

    A candidate is dropped if it is within ``threshold`` bits of any frame
    already kept. Pure function so the policy is unit-testable.
    """
    kept_indices: list[int] = []
    kept_hashes: list[imagehash.ImageHash] = []
    for idx, h in enumerate(hashes):
        if any((h - kept) <= threshold for kept in kept_hashes):
            continue
        kept_indices.append(idx)
        kept_hashes.append(h)
    return kept_indices


def _phash(
    image_path: Path,
    mask_regions: list[tuple[float, float, float, float]] | None = None,
) -> imagehash.ImageHash:
    """Perceptual hash of a frame, with any masked regions blanked first."""
    with Image.open(image_path) as img:
        frame = img.convert("RGB")
        if mask_regions:
            width, height = frame.size
            draw = ImageDraw.Draw(frame)
            for x0, y0, x1, y1 in mask_regions:
                draw.rectangle(
                    [
                        int(x0 * width),
                        int(y0 * height),
                        int(x1 * width),
                        int(y1 * height),
                    ],
                    fill=(255, 255, 255),
                )
        return imagehash.phash(frame, hash_size=PHASH_SIZE)


def _extract_frame(video_path: Path, timestamp: float, out_path: Path) -> None:
    """Grab a single JPEG frame at ``timestamp`` with ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        # mjpeg expects full-range YUV; force it so limited-range (tv) sources
        # don't crash the encoder ("Non full-range YUV is non-standard").
        "-pix_fmt",
        "yuvj420p",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists():
        raise VideoDistillError(
            f"ffmpeg failed to extract a frame at {timestamp:.3f}s from "
            f"{video_path}:\n{result.stderr.strip()}"
        )
