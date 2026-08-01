"""Shared exception types. Importable by any stage (it is not itself a stage)."""

from __future__ import annotations


class VideoDistillError(Exception):
    """Base class for all VideoDistill errors."""


class StageNotImplemented(VideoDistillError):
    """Raised by stub stages that have not been built yet."""


class DependencyMissing(VideoDistillError):
    """A required external tool (e.g. ffmpeg) is not available."""


class ProfileError(VideoDistillError):
    """A domain profile is missing, unreadable, or invalid."""


class ProviderError(VideoDistillError):
    """An LLM provider could not be constructed or used (e.g. missing key)."""


class CostLimitExceeded(VideoDistillError):
    """An estimated LLM spend exceeds the caller's --max-cost guard."""
