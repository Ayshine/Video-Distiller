"""Tests for the extract_visuals stage.

The provider is mocked (no network, no key). Covers JSON validation, the
retry-once-then-skip path, the cost guard, the cost formula, and the
identical-frame cache-hit path via CachedLLMClient.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from videodistill.errors import CostLimitExceeded
from videodistill.llm.cache import CachedLLMClient
from videodistill.models import Keyframe, KeyframeSet, VisualKind
from videodistill.profile import DomainProfile
from videodistill.stages import extract_visuals

PROFILE = DomainProfile(name="generic")

VALID_REPLY = json.dumps(
    {
        "kind": "code",
        "text": "def main():\n    return 0",
        "code_language": "python",
        "description": "a minimal main function",
    }
)


class _FakeVision:
    """Returns queued replies; counts vision calls."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def vision(
        self,
        prompt: str,
        image_path: Path,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        self.calls += 1
        return self._replies.pop(0)


def _keyframe(
    tmp_path: Path,
    name: str = "k0.jpg",
    ts: float = 1.0,
    content: bytes = b"img-bytes",
    on_disk: bool = True,
) -> Keyframe:
    path = tmp_path / name
    if on_disk:
        path.write_bytes(content)
    return Keyframe(timestamp=ts, image_path=str(path), phash="0" * 16)


def test_valid_json_produces_extract(tmp_path: Path) -> None:
    kf = _keyframe(tmp_path)
    provider = _FakeVision([VALID_REPLY])

    result = extract_visuals.run(
        KeyframeSet(keyframes=[kf]), tmp_path, PROFILE, llm=provider, model="gpt-4o"
    )

    assert len(result.extracts) == 1
    extract = result.extracts[0]
    assert extract.kind == VisualKind.code
    assert extract.code_language == "python"
    assert extract.timestamp == 1.0  # injected from the keyframe, not the model
    assert "def main()" in extract.text
    assert provider.calls == 1
    assert (tmp_path / "visuals.json").exists()


def test_markdown_fenced_json_is_accepted(tmp_path: Path) -> None:
    kf = _keyframe(tmp_path)
    fenced = f"```json\n{VALID_REPLY}\n```"
    provider = _FakeVision([fenced])

    result = extract_visuals.run(
        KeyframeSet(keyframes=[kf]), tmp_path, PROFILE, llm=provider, model="gpt-4o"
    )
    assert len(result.extracts) == 1


def test_retry_once_then_succeeds(tmp_path: Path) -> None:
    kf = _keyframe(tmp_path)
    provider = _FakeVision(["not valid json at all", VALID_REPLY])

    result = extract_visuals.run(
        KeyframeSet(keyframes=[kf]), tmp_path, PROFILE, llm=provider, model="gpt-4o"
    )

    assert len(result.extracts) == 1
    assert provider.calls == 2  # first failed, retry succeeded


def test_failure_after_retry_is_skipped_not_crashed(tmp_path: Path) -> None:
    kf = _keyframe(tmp_path)
    provider = _FakeVision(["garbage", "still garbage"])

    result = extract_visuals.run(
        KeyframeSet(keyframes=[kf]), tmp_path, PROFILE, llm=provider, model="gpt-4o"
    )

    assert result.extracts == []  # frame marked failed, pipeline survives
    assert provider.calls == 2


def test_one_bad_frame_does_not_sink_the_good_ones(tmp_path: Path) -> None:
    good = _keyframe(tmp_path, "good.jpg", ts=1.0)
    bad = _keyframe(tmp_path, "bad.jpg", ts=2.0)
    # good: valid immediately; bad: invalid twice (retry then fail).
    provider = _FakeVision([VALID_REPLY, "nope", "nope again"])

    result = extract_visuals.run(
        KeyframeSet(keyframes=[good, bad]),
        tmp_path,
        PROFILE,
        llm=provider,
        model="gpt-4o",
    )

    assert len(result.extracts) == 1
    assert result.extracts[0].timestamp == 1.0


class _RaisingVision:
    """Simulates a vision call that errors (e.g. a rate limit)."""

    def vision(
        self,
        prompt: str,
        image_path: Path,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        raise RuntimeError("429 rate limit")


def test_vision_api_error_skips_frame_without_crashing(tmp_path: Path) -> None:
    kf = _keyframe(tmp_path)
    result = extract_visuals.run(
        KeyframeSet(keyframes=[kf]),
        tmp_path,
        PROFILE,
        llm=_RaisingVision(),
        model="gpt-4o",
    )
    assert result.extracts == []  # frame skipped, run survives


def test_cost_guard_blocks_before_any_call(tmp_path: Path) -> None:
    # 300 frames x $0.01 = $3.00 > default $2.00.
    frames = [_keyframe(tmp_path, f"k{i}.jpg", on_disk=False) for i in range(300)]
    provider = _FakeVision([])

    with pytest.raises(CostLimitExceeded, match=r"\$3\.00"):
        extract_visuals.run(
            KeyframeSet(keyframes=frames),
            tmp_path,
            PROFILE,
            llm=provider,
            model="gpt-4o",
            max_cost=2.00,
        )
    assert provider.calls == 0  # aborted before touching the provider


def test_estimate_cost_formula() -> None:
    assert extract_visuals.estimate_cost(0) == 0.0
    assert extract_visuals.estimate_cost(10) == pytest.approx(0.10)
    assert extract_visuals.estimate_cost(200) == pytest.approx(2.00)


def test_identical_frames_are_cached(tmp_path: Path) -> None:
    # Same bytes across two runs -> the second run is fully served from cache.
    kf = _keyframe(tmp_path, "frame.jpg", content=b"identical-bytes")
    provider = _FakeVision([VALID_REPLY, VALID_REPLY])
    cached = CachedLLMClient(provider, tmp_path / "cache")

    first = extract_visuals.run(
        KeyframeSet(keyframes=[kf]), tmp_path, PROFILE, llm=cached, model="gpt-4o"
    )
    second = extract_visuals.run(
        KeyframeSet(keyframes=[kf]), tmp_path, PROFILE, llm=cached, model="gpt-4o"
    )

    assert len(first.extracts) == 1 and len(second.extracts) == 1
    assert provider.calls == 1  # second run hit the content-hash cache


def test_prompt_prefers_profile_code_languages() -> None:
    profile = DomainProfile(name="sample", code_languages=["python", "rust"])
    prompt = extract_visuals._build_prompt(profile)
    assert "python, rust" in prompt  # profile.code_languages surfaced to the model
    generic_prompt = extract_visuals._build_prompt(PROFILE)
    assert "python, rust" not in generic_prompt


def _write_png(path: Path, width: int, height: int) -> Path:
    Image.new("RGB", (width, height), "white").save(path)
    return path


def test_prepare_image_downscales_wide_frame(tmp_path: Path) -> None:
    src = _write_png(tmp_path / "wide.png", 1920, 1080)
    out, is_temp = extract_visuals._prepare_image(src, 1024)
    assert is_temp and out != src
    with Image.open(out) as img:
        assert img.width == 1024 and img.height == 576  # aspect preserved
    out.unlink()


def test_prepare_image_leaves_small_or_disabled_untouched(tmp_path: Path) -> None:
    small = _write_png(tmp_path / "small.png", 800, 600)
    assert extract_visuals._prepare_image(small, 1024) == (small, False)  # already ≤
    wide = _write_png(tmp_path / "w.png", 1920, 1080)
    assert extract_visuals._prepare_image(wide, None) == (wide, False)  # disabled


def test_run_downscales_and_cleans_up_temp(tmp_path: Path) -> None:
    _write_png(tmp_path / "frame.jpg", 1920, 1080)
    kf = Keyframe(timestamp=1.0, image_path=str(tmp_path / "frame.jpg"), phash="0" * 16)

    seen_width: list[int] = []
    seen_path: list[Path] = []

    class _SizeSpy:
        calls = 0

        def vision(
            self, prompt: str, image_path: Path, *, model: str, **kw: object
        ) -> str:
            self.calls += 1
            with Image.open(image_path) as img:
                seen_width.append(img.width)
            seen_path.append(Path(image_path))
            return VALID_REPLY

    result = extract_visuals.run(
        KeyframeSet(keyframes=[kf]),
        tmp_path,
        PROFILE,
        llm=_SizeSpy(),
        model="gpt-4o",
        image_max_width=1024,
    )
    assert len(result.extracts) == 1
    assert seen_width == [1024]  # the provider saw the downscaled image
    # The exact temp frame this test created was cleaned up. (Check that one
    # path, not a global tempdir glob — which would flake against concurrent runs.)
    tmp_frame = seen_path[0]
    assert tmp_frame.name.startswith("vd_frame_")  # it was downscaled to a temp file
    assert not tmp_frame.exists()  # and removed after the call
