"""Evaluation harness.

One command — ``videodistill eval --job <dir>`` — turns a completed job's
artifacts into a single markdown report: does extracted code compile, did the
ASR catch the domain vocabulary, are the notes grounded, and what did the run
cost / compress to. The numbers are meant to go straight into the README.
"""

from videodistill.evals.report import evaluate

__all__ = ["evaluate"]
