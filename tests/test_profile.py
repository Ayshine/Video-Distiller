"""Tests for the domain profile loader.

Covers the shipped generic profile, loading a domain profile from YAML, invalid
YAML / schema, and the empty-vocabulary ASR-prompt path — the behaviours that
keep domain knowledge in YAML and out of the code. Domain profiles are written
to a temp dir so the suite never depends on any specific profile shipping.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videodistill.errors import ProfileError
from videodistill.profile import (
    DomainProfile,
    available_profiles,
    load_profile,
)

# A throwaway domain profile, written to a temp dir per test. Deliberately not
# any real subject — it just exercises the loader and the ASR-prompt builder.
SAMPLE_PROFILE_YAML = """
name: sample
description: A sample technical domain.
vocabulary:
  - widget
  - flux capacitor
  - foo::bar
code_languages:
  - python
concept_id_prefix: smp
verification:
  kind: compile_check
  command: "python -m py_compile {file}"
distill_hints: Prefer the speaker's exact terminology.
"""


def _write_sample(dir_path: Path) -> Path:
    (dir_path / "sample.yaml").write_text(SAMPLE_PROFILE_YAML, encoding="utf-8")
    return dir_path


def test_generic_profile_is_domain_free() -> None:
    profile = load_profile("generic")
    assert profile.name == "generic"
    assert profile.vocabulary == []
    assert profile.verification is None
    # Empty vocabulary => no ASR bias at all.
    assert profile.asr_initial_prompt() is None


def test_domain_profile_loads_from_yaml(tmp_path: Path) -> None:
    profile = load_profile("sample", profiles_dir=_write_sample(tmp_path))
    assert profile.name == "sample"
    assert profile.concept_id_prefix == "smp"
    assert "python" in profile.code_languages
    assert profile.verification is not None
    assert profile.verification.kind == "compile_check"
    assert "{file}" in profile.verification.command
    # A multi-word term survived the YAML round-trip.
    assert "flux capacitor" in profile.vocabulary


def test_domain_asr_prompt_includes_vocabulary(tmp_path: Path) -> None:
    profile = load_profile("sample", profiles_dir=_write_sample(tmp_path))
    prompt = profile.asr_initial_prompt()
    assert prompt is not None
    assert "foo::bar" in prompt  # jargon seeded into the decoder


def test_generic_profile_is_discoverable() -> None:
    # The generic profile always ships; no domain profile is assumed present.
    assert "generic" in available_profiles()


def test_missing_profile_raises_with_hint() -> None:
    with pytest.raises(ProfileError, match="not found"):
        load_profile("does-not-exist")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text("name: broken\nvocabulary: [unterminated\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="not valid YAML"):
        load_profile("broken", profiles_dir=tmp_path)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "typo.yaml"
    bad.write_text("name: typo\nvocabullary: []\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="invalid"):
        load_profile("typo", profiles_dir=tmp_path)


def test_non_mapping_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="mapping"):
        load_profile("list", profiles_dir=tmp_path)


def test_empty_vocabulary_profile_has_no_prompt() -> None:
    profile = DomainProfile(name="x", vocabulary=[])
    assert profile.asr_initial_prompt() is None
