"""Tests for the align stage.

Each policy rule is exercised in isolation with hand-built artifacts (no video):
anchoring, look-ahead move, scroll threshold at 59% vs 61%, orphan attach, and
the 5-minute pause split. A property test asserts every segment lands in exactly
one chunk.
"""

from __future__ import annotations

from videodistill.models import (
    Transcript,
    TranscriptSegment,
    VisualExtract,
    VisualExtractSet,
    VisualKind,
)
from videodistill.stages.align import (
    FLAG_MERGED_SCROLL,
    FLAG_ORPHAN,
    _align,
    _line_share,
    _make_chunk,
    _merge_scrolls,
)


def _seg(start: float, end: float, text: str = "") -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def _vis(ts: float, text: str, kind: VisualKind = VisualKind.slide) -> VisualExtract:
    return VisualExtract(timestamp=ts, kind=kind, text=text)


def _timeline(
    segments: list[TranscriptSegment],
    visuals: list[VisualExtract],
    language: str = "en",
):
    return _align(
        Transcript(language=language, model_size="small", segments=segments),
        VisualExtractSet(extracts=visuals),
    )


# --- Rule 1: anchoring ------------------------------------------------------


def test_segments_anchor_to_most_overlapping_interval() -> None:
    v0 = _vis(0.0, "intro")
    v1 = _vis(20.0, "details")
    # a: fully in [0,20); b: fully in [20,inf); c: before the first visual -> v0
    a = _seg(2, 5, "alpha")
    b = _seg(25, 30, "bravo")
    c = _seg(-5, -1, "charlie")
    timeline, report = _timeline([a, b, c], [v0, v1])

    assert report.chunk_count == 2
    assert [s.text for s in timeline.chunks[0].segments] == ["charlie", "alpha"]
    assert [s.text for s in timeline.chunks[1].segments] == ["bravo"]


# --- Rule 2: look-ahead -----------------------------------------------------


def test_lookahead_moves_late_segment_to_next_visual() -> None:
    v0 = _vis(0.0, "introduction overview agenda")
    v1 = _vis(20.0, "pointers memory address dereference")
    # In interval 0 but in its last 10s [10,20); matches v1's words, not v0's.
    late = _seg(12, 14, "now about pointers and memory")
    early = _seg(2, 5, "welcome to the overview")
    timeline, report = _timeline([early, late], [v0, v1])

    assert report.lookahead_moves == 1
    assert [s.text for s in timeline.chunks[0].segments] == ["welcome to the overview"]
    assert [s.text for s in timeline.chunks[1].segments] == [
        "now about pointers and memory"
    ]


def test_lookahead_uses_language_specific_stopwords_turkish() -> None:
    # Turkish transcript: stopwords (şimdi/ve) are filtered via languages/tr,
    # and the Unicode tokenizer keeps "işaretçi"/"bellek" intact.
    v0 = _vis(0.0, "giriş genel bakış")  # intro / general / overview
    v1 = _vis(20.0, "işaretçi bellek adres")  # pointer / memory / address
    late = _seg(12, 14, "şimdi işaretçi ve bellek")  # now pointer and memory
    early = _seg(2, 5, "giriş bölümü")
    timeline, report = _timeline([early, late], [v0, v1], language="tr")

    assert report.lookahead_moves == 1
    assert timeline.chunks[1].segments[0].text == "şimdi işaretçi ve bellek"


def test_lookahead_does_not_move_when_not_in_window() -> None:
    v0 = _vis(0.0, "introduction overview")
    v1 = _vis(20.0, "pointers memory")
    # Matches v1 lexically but is early in interval 0 (not the last 10s).
    seg = _seg(2, 4, "pointers memory")
    _, report = _timeline([seg], [v0, v1])
    assert report.lookahead_moves == 0


# --- Rule 3: scroll merge ---------------------------------------------------


def _code_block(lines: list[str]) -> str:
    return "\n".join(lines)


def test_line_share_ratio_straddles_threshold() -> None:
    base = [f"line{i}" for i in range(100)]
    # Shares 61 of base's lines (in order), then 39 novel lines -> 61/100.
    b61 = base[39:100] + [f"new{i}" for i in range(39)]
    # Shares 59 -> 59/100.
    b59 = base[41:100] + [f"new{i}" for i in range(41)]

    assert _line_share(_code_block(base), _code_block(b61)) == 0.61
    assert _line_share(_code_block(base), _code_block(b59)) == 0.59


def test_scroll_merges_at_61_not_59() -> None:
    base = [f"line{i}" for i in range(100)]
    b61 = base[39:100] + [f"new{i}" for i in range(39)]
    b59 = base[41:100] + [f"new{i}" for i in range(41)]

    def code_chunk(ts: float, lines: list[str]):
        return _make_chunk([_vis(ts, _code_block(lines), VisualKind.code)], [])

    merged, merges = _merge_scrolls([code_chunk(0, base), code_chunk(10, b61)])
    assert merges == 1
    assert len(merged) == 1
    assert FLAG_MERGED_SCROLL in merged[0].flags
    assert merged[0].visuals[0].timestamp == 0.0  # earliest kept

    not_merged, merges2 = _merge_scrolls([code_chunk(0, base), code_chunk(10, b59)])
    assert merges2 == 0
    assert len(not_merged) == 2


def test_scroll_merge_collapses_duplicate_lines() -> None:
    a = _make_chunk([_vis(0, "l1\nl2\nl3", VisualKind.code)], [])
    b = _make_chunk([_vis(5, "l2\nl3\nl4", VisualKind.code)], [])  # 100% of min shared
    merged, merges = _merge_scrolls([a, b])
    assert merges == 1
    assert merged[0].visuals[0].text == "l1\nl2\nl3\nl4"  # union, order preserved


# --- Rule 4: orphan attach --------------------------------------------------


def test_orphan_visual_attaches_to_following_chunk() -> None:
    v0 = _vis(5.0, "a slide nobody talks over yet")  # no overlapping speech
    v1 = _vis(50.0, "the discussed slide")
    seg = _seg(51, 55, "talking about the discussed slide")
    timeline, report = _timeline([seg], [v0, v1])

    assert report.orphans == 1
    assert len(timeline.chunks) == 1
    chunk = timeline.chunks[0]
    assert FLAG_ORPHAN in chunk.flags
    # The orphan visual rode along into the following chunk.
    assert [v.timestamp for v in chunk.visuals] == [5.0, 50.0]
    assert len(chunk.segments) == 1


# --- Rule 5: bounds split ---------------------------------------------------


def test_long_chunk_splits_at_largest_pause() -> None:
    v0 = _vis(0.0, "one slide for the whole talk")
    # Span 0..320 (> 300s) with a single huge pause between 50 and 250.
    segments = [
        _seg(0, 10),
        _seg(20, 30),
        _seg(40, 50),
        _seg(250, 260),
        _seg(270, 280),
        _seg(290, 300),
        _seg(310, 320),
    ]
    timeline, _ = _timeline(segments, [v0])

    assert len(timeline.chunks) == 2
    assert timeline.chunks[0].segments[-1].end == 50.0
    assert timeline.chunks[1].segments[0].start == 250.0
    # The visual stays with the side that contains its timestamp.
    assert timeline.chunks[0].visuals[0].timestamp == 0.0
    assert timeline.chunks[1].visuals == []


def test_short_chunk_is_not_split() -> None:
    v0 = _vis(0.0, "slide")
    segments = [_seg(0, 10), _seg(20, 30), _seg(40, 50)]  # span 50s
    timeline, _ = _timeline(segments, [v0])
    assert len(timeline.chunks) == 1


# --- Property test ----------------------------------------------------------


def test_every_segment_lands_in_exactly_one_chunk() -> None:
    visuals = [
        _vis(0.0, "introduction overview"),
        _vis(30.0, "vector container iterator", VisualKind.code),
        _vis(60.0, "vector container push back", VisualKind.code),  # scroll-ish
        _vis(400.0, "summary conclusion"),
    ]
    segments = [
        _seg(-3, -1, "before anything starts"),
        _seg(5, 8, "welcome to the overview"),
        _seg(28, 29, "now the vector container"),  # look-ahead candidate
        _seg(35, 40, "iterating the vector"),
        _seg(65, 70, "push back onto the vector"),
        _seg(405, 410, "in summary"),
        # A long tail forcing a bounds split on the last chunk:
        _seg(420, 430),
        _seg(700, 710),
        _seg(720, 730),
    ]
    timeline, report = _timeline(segments, visuals)

    collected = [s for chunk in timeline.chunks for s in chunk.segments]
    assert len(collected) == len(segments)  # exactly one chunk each, none lost
    assert sorted(s.start for s in collected) == sorted(s.start for s in segments)
    assert report.segment_count == len(segments)


def test_no_visuals_still_produces_chunks() -> None:
    segments = [_seg(0, 10, "hello"), _seg(10, 20, "world")]
    timeline, report = _timeline(segments, [])
    assert report.chunk_count == 1
    assert len(timeline.chunks[0].segments) == 2
    assert timeline.chunks[0].visuals == []
