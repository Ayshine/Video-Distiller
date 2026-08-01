"""Stage 5 — align.

Bind VisualExtracts to TranscriptSegments into AlignedChunks — "what was shown"
next to "what was said". Implements this fixed policy (no invented heuristics):

1. Anchor on visuals: each keyframe defines interval [ts, next_ts). A segment
   belongs to the interval it overlaps most by duration.
2. Look-ahead: a segment in the last 10s of interval N moves to N+1 if its
   lexical overlap (shared content words) with N+1's visual text exceeds its
   overlap with N's.
3. Scroll merge: consecutive code intervals sharing >= 60% of lines
   (order-preserving) are one scrolled document -> merge, collapse duplicate
   lines, keep the earliest timestamp, flag ``merged_scroll``.
4. Orphan visuals (no overlapping speech) attach to the following chunk, which
   is flagged ``orphan``.
5. Bounds: a chunk over 5 min of transcript splits at its largest inter-segment
   pause (never mid-sentence).

Writes :class:`~videodistill.models.AlignedTimeline` and an
:class:`~videodistill.models.AlignmentReport`. Domain-independent; the profile
is accepted for a uniform stage signature.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection
from pathlib import Path

from videodistill.language import load_stopwords
from videodistill.models import (
    AlignedChunk,
    AlignedTimeline,
    AlignmentReport,
    Transcript,
    TranscriptSegment,
    VisualExtract,
    VisualExtractSet,
    VisualKind,
)
from videodistill.profile import DomainProfile

IS_STUB = False

logger = logging.getLogger("videodistill.stages.align")

# Policy constants (from the spec; do not tune silently).
LOOKAHEAD_WINDOW_S = 10.0
SCROLL_LINE_SHARE_THRESHOLD = 0.60
MAX_CHUNK_TRANSCRIPT_S = 300.0  # 5 minutes

FLAG_MERGED_SCROLL = "merged_scroll"
FLAG_ORPHAN = "orphan"

# Unicode word tokenizer: \w keeps letters of any language (Turkish ç/ğ/ı/ş/ü,
# etc.), unlike an ASCII-only [a-z] class which would split those words apart.
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def run(
    transcript: Transcript,
    visuals: VisualExtractSet,
    job_dir: Path,
    profile: DomainProfile,
) -> AlignedTimeline:
    """Align a transcript and its visuals into chunks; write both artifacts."""
    timeline, report = _align(transcript, visuals)
    timeline.save(job_dir)
    report.save(job_dir)
    logger.info(
        "align: %d chunk(s) | look-ahead moves=%d, scroll merges=%d, orphans=%d",
        report.chunk_count,
        report.lookahead_moves,
        report.scroll_merges,
        report.orphans,
    )
    return timeline


def _align(
    transcript: Transcript, visuals: VisualExtractSet
) -> tuple[AlignedTimeline, AlignmentReport]:
    """Pure alignment (no I/O) so each rule is unit-testable."""
    segments = sorted(transcript.segments, key=lambda s: s.start)
    vis = sorted(visuals.extracts, key=lambda v: v.timestamp)

    if not vis:
        # No visuals to anchor on: one chunk of all speech, then bounds-split.
        chunks = _bounds_split_all([_make_chunk([], segments)]) if segments else []
        return (
            AlignedTimeline(chunks=chunks),
            AlignmentReport(chunk_count=len(chunks), segment_count=len(segments)),
        )

    # Stopwords for the spoken language (detected by transcribe). Unknown
    # languages get an empty set, which simply keeps filler words.
    stopwords = load_stopwords(transcript.language)

    # Rule 1 — anchor segments on visual intervals.
    assignments = _assign_segments(segments, vis)

    # Rule 2 — look-ahead moves.
    lookahead_moves = _apply_lookahead(assignments, vis, stopwords)

    # One chunk per visual, in visual order.
    chunks = [_make_chunk([vis[i]], assignments[i]) for i in range(len(vis))]

    # Rule 3 — scroll merge.
    chunks, scroll_merges = _merge_scrolls(chunks)

    # Rule 4 — orphan attach.
    chunks, orphans = _attach_orphans(chunks)

    # Rule 5 — bounds split.
    chunks = _bounds_split_all(chunks)

    report = AlignmentReport(
        chunk_count=len(chunks),
        segment_count=len(segments),
        lookahead_moves=lookahead_moves,
        scroll_merges=scroll_merges,
        orphans=orphans,
    )
    return AlignedTimeline(chunks=chunks), report


# --- Rule 1: anchoring ------------------------------------------------------


def _intervals(vis: list[VisualExtract]) -> list[tuple[float, float]]:
    """[ts, next_ts) per visual; the last extends to +inf."""
    bounds: list[tuple[float, float]] = []
    for i, v in enumerate(vis):
        end = vis[i + 1].timestamp if i + 1 < len(vis) else float("inf")
        bounds.append((v.timestamp, end))
    return bounds


def _assign_segments(
    segments: list[TranscriptSegment], vis: list[VisualExtract]
) -> list[list[TranscriptSegment]]:
    """Assign each segment to the interval it overlaps most (by duration).

    A segment entirely before the first visual has zero overlap everywhere and
    falls to interval 0 (the earliest), which the ``>`` comparison guarantees.
    """
    bounds = _intervals(vis)
    assignments: list[list[TranscriptSegment]] = [[] for _ in vis]
    for seg in segments:
        best_i, best_overlap = 0, -1.0
        for i, (start, end) in enumerate(bounds):
            overlap = max(0.0, min(seg.end, end) - max(seg.start, start))
            if overlap > best_overlap:
                best_overlap, best_i = overlap, i
        assignments[best_i].append(seg)
    return assignments


# --- Rule 2: look-ahead -----------------------------------------------------


def _apply_lookahead(
    assignments: list[list[TranscriptSegment]],
    vis: list[VisualExtract],
    stopwords: Collection[str],
) -> int:
    """Move late segments of interval N to N+1 when they match N+1's visual.

    Processes left to right; a moved segment may cascade further forward.
    """
    bounds = _intervals(vis)
    moves = 0
    for i in range(len(vis) - 1):
        boundary = bounds[i][1]  # == vis[i + 1].timestamp
        window_start = boundary - LOOKAHEAD_WINDOW_S
        staying: list[TranscriptSegment] = []
        for seg in assignments[i]:
            in_window = window_start <= seg.start < boundary
            if in_window:
                overlap_here = _lexical_overlap(seg.text, vis[i].text, stopwords)
                overlap_next = _lexical_overlap(seg.text, vis[i + 1].text, stopwords)
                if overlap_next > overlap_here:
                    assignments[i + 1].append(seg)
                    moves += 1
                    continue
            staying.append(seg)
        assignments[i] = staying

    for group in assignments:
        group.sort(key=lambda s: s.start)
    return moves


def _content_words(text: str, stopwords: Collection[str]) -> set[str]:
    return {
        w for w in _WORD_RE.findall(text.lower()) if len(w) >= 2 and w not in stopwords
    }


def _lexical_overlap(a: str, b: str, stopwords: Collection[str]) -> int:
    """Number of shared content words between two texts."""
    return len(_content_words(a, stopwords) & _content_words(b, stopwords))


# --- Rule 3: scroll merge ---------------------------------------------------


def _is_code_chunk(chunk: AlignedChunk) -> bool:
    return len(chunk.visuals) == 1 and chunk.visuals[0].kind == VisualKind.code


def _merge_scrolls(chunks: list[AlignedChunk]) -> tuple[list[AlignedChunk], int]:
    """Merge runs of consecutive code chunks that are one scrolled document."""
    result: list[AlignedChunk] = []
    merges = 0
    i = 0
    while i < len(chunks):
        current = chunks[i]
        j = i + 1
        while (
            j < len(chunks)
            and _is_code_chunk(current)
            and _is_code_chunk(chunks[j])
            and _line_share(current.visuals[0].text, chunks[j].visuals[0].text)
            >= SCROLL_LINE_SHARE_THRESHOLD
        ):
            current = _merge_two_code_chunks(current, chunks[j])
            merges += 1
            j += 1
        result.append(current)
        i = j
    return result, merges


def _lines(text: str) -> list[str]:
    return [ln.rstrip() for ln in text.splitlines()]


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two line lists."""
    prev = [0] * (len(b) + 1)
    for line_a in a:
        curr = [0] * (len(b) + 1)
        for j, line_b in enumerate(b, start=1):
            if line_a == line_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def _line_share(a: str, b: str) -> float:
    """Order-preserving shared-line fraction, relative to the shorter block."""
    lines_a, lines_b = _lines(a), _lines(b)
    if not lines_a or not lines_b:
        return 0.0
    return _lcs_length(lines_a, lines_b) / min(len(lines_a), len(lines_b))


def _merge_two_code_chunks(a: AlignedChunk, b: AlignedChunk) -> AlignedChunk:
    """Combine two scrolled code chunks into one (duplicate lines collapsed)."""
    va, vb = a.visuals[0], b.visuals[0]
    merged_lines = _dedup_preserve_order(va.text.splitlines() + vb.text.splitlines())
    merged_visual = VisualExtract(
        timestamp=min(va.timestamp, vb.timestamp),  # keep earliest
        kind=VisualKind.code,
        text="\n".join(merged_lines),
        code_language=va.code_language or vb.code_language,
        description=va.description or vb.description,
    )
    segments = sorted(a.segments + b.segments, key=lambda s: s.start)
    flags = list(dict.fromkeys([*a.flags, *b.flags, FLAG_MERGED_SCROLL]))
    return _make_chunk([merged_visual], segments, flags=flags)


def _dedup_preserve_order(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.rstrip()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


# --- Rule 4: orphan attach --------------------------------------------------


def _attach_orphans(chunks: list[AlignedChunk]) -> tuple[list[AlignedChunk], int]:
    """Attach visuals with no overlapping speech to the following chunk."""
    result: list[AlignedChunk] = []
    pending: list[VisualExtract] = []
    orphans = 0
    for chunk in chunks:
        if not chunk.segments:
            pending.extend(chunk.visuals)
            orphans += 1
            continue
        if pending:
            visuals = pending + chunk.visuals
            flags = list(dict.fromkeys([*chunk.flags, FLAG_ORPHAN]))
            chunk = _make_chunk(visuals, chunk.segments, flags=flags)
            pending = []
        result.append(chunk)

    if pending:
        # Orphans at the very end have no following chunk to attach to.
        result.append(_make_chunk(pending, [], flags=[FLAG_ORPHAN]))
    return result, orphans


# --- Rule 5: bounds split ---------------------------------------------------


def _bounds_split_all(chunks: list[AlignedChunk]) -> list[AlignedChunk]:
    out: list[AlignedChunk] = []
    for chunk in chunks:
        out.extend(_split_chunk(chunk))
    return out


def _split_chunk(chunk: AlignedChunk) -> list[AlignedChunk]:
    """Recursively split a chunk over 5 min at its largest inter-segment pause."""
    segments = chunk.segments
    if len(segments) < 2:
        return [chunk]
    span = segments[-1].end - segments[0].start
    if span <= MAX_CHUNK_TRANSCRIPT_S:
        return [chunk]

    # Largest pause between consecutive segments (earliest one on ties).
    best_k, best_gap = 0, -1.0
    for k in range(len(segments) - 1):
        gap = segments[k + 1].start - segments[k].end
        if gap > best_gap:
            best_gap, best_k = gap, k

    left_segs = segments[: best_k + 1]
    right_segs = segments[best_k + 1 :]
    split_point = right_segs[0].start

    left_vis = [v for v in chunk.visuals if v.timestamp < split_point]
    right_vis = [v for v in chunk.visuals if v.timestamp >= split_point]

    left = _make_chunk(left_vis, left_segs, flags=list(chunk.flags))
    right = _make_chunk(right_vis, right_segs, flags=list(chunk.flags))
    return _split_chunk(left) + _split_chunk(right)


# --- chunk construction -----------------------------------------------------


def _make_chunk(
    visuals: list[VisualExtract],
    segments: list[TranscriptSegment],
    *,
    flags: list[str] | None = None,
) -> AlignedChunk:
    """Build a chunk, deriving its time bounds from its contents."""
    starts = [s.start for s in segments] + [v.timestamp for v in visuals]
    ends = [s.end for s in segments] + [v.timestamp for v in visuals]
    start = min(starts) if starts else 0.0
    end = max(ends) if ends else start
    return AlignedChunk(
        start=start,
        end=end,
        segments=sorted(segments, key=lambda s: s.start),
        visuals=visuals,
        flags=flags or [],
    )
