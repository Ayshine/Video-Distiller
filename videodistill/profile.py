"""Domain profiles.

A :class:`DomainProfile` is the ONLY place domain knowledge lives. Stages take
a profile and read behaviour from it; they never hardcode a domain term or a
video-type assumption. Profiles are YAML files in ``profiles/<name>.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from videodistill.errors import ProfileError

# profiles/ lives at the repo root, next to the videodistill package.
_PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


class Verification(BaseModel):
    """How to verify extracted code for a domain (e.g. a compile check).

    ``command`` uses ``{file}`` as a placeholder for a snippet written to a
    temporary file. Consumed by the eval harness in a later session.
    """

    kind: str
    command: str


class DomainProfile(BaseModel):
    """Everything a domain contributes to the pipeline.

    Kept permissive on purpose: a profile is data, and new fields should be
    additive. ``model_config`` forbids unknown keys so typos in a YAML file
    surface as errors rather than being silently ignored.
    """

    model_config = {"extra": "forbid"}

    name: str
    description: str = ""
    vocabulary: list[str] = Field(default_factory=list)
    code_languages: list[str] = Field(default_factory=list)
    # Spoken language of the audio as an ISO-639-1 code (e.g. "tr"). When set,
    # ASR is told the language instead of auto-detecting per chunk — this stops
    # a mis-detected chunk (e.g. a quiet intro read as English) from silently
    # dropping content. None = let the model auto-detect. A property of the
    # source material, so it lives in the profile, not the pipeline.
    asr_language: str | None = None
    concept_id_prefix: str = "gen"
    verification: Verification | None = None
    distill_hints: str = ""
    # Rectangles (x0, y0, x1, y1 as 0..1 fractions of width/height) blanked
    # before perceptual hashing, so a moving overlay (e.g. a webcam) does not
    # trigger or mask keyframes. A video-layout assumption — hence in the
    # profile, not the pipeline code. Empty = hash the whole frame.
    keyframe_mask_regions: list[tuple[float, float, float, float]] = Field(
        default_factory=list
    )

    def asr_initial_prompt(self) -> str | None:
        """Build the faster-whisper initial prompt from the vocabulary.

        Returns ``None`` when the vocabulary is empty, so the generic profile
        biases ASR toward nothing.
        """
        if not self.vocabulary:
            return None
        terms = ", ".join(self.vocabulary)
        subject = self.description or self.name
        return (
            f"The following is a transcript about {subject}. "
            f"Expect terminology such as: {terms}."
        )


def available_profiles() -> list[str]:
    """Names of the profiles shipped in the profiles/ directory."""
    if not _PROFILES_DIR.is_dir():
        return []
    return sorted(p.stem for p in _PROFILES_DIR.glob("*.yaml"))


def load_profile(name: str, profiles_dir: Path | None = None) -> DomainProfile:
    """Load and validate ``profiles/<name>.yaml``.

    Raises :class:`ProfileError` with an actionable message when the file is
    missing, is not valid YAML, or does not match the profile schema.
    """
    directory = profiles_dir or _PROFILES_DIR
    path = directory / f"{name}.yaml"
    if not path.exists():
        known = ", ".join(available_profiles()) or "(none found)"
        raise ProfileError(
            f"Profile '{name}' not found at {path}. Available profiles: {known}."
        )

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProfileError(f"Profile '{name}' is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ProfileError(
            f"Profile '{name}' must be a YAML mapping, got {type(raw).__name__}."
        )

    try:
        return DomainProfile.model_validate(raw)
    except ValidationError as exc:
        raise ProfileError(f"Profile '{name}' is invalid:\n{exc}") from exc
