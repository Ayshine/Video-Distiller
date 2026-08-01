"""Typer CLI entry point.

Session 0 commands:
  - ``process`` runs the whole pipeline over a video with a domain profile.
  - ``run-stage`` re-runs a single stage against an existing job directory.

Later sessions add eval / kb.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from videodistill.config import Config, load_config
from videodistill.errors import StageNotImplemented, VideoDistillError
from videodistill.llm.base import LLMClient
from videodistill.pipeline import STAGE_ORDER, run_single_stage
from videodistill.profile import load_profile
from videodistill.stages.extract_visuals import DEFAULT_MAX_COST_USD

app = typer.Typer(
    help="VideoDistill — distill any video into a reviewable digest.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _main() -> None:
    """Load .env (if present) before any command, so OPENAI_API_KEY etc. resolve.

    Real environment variables already set are left untouched (override=False).
    """
    from dotenv import load_dotenv

    load_dotenv()


def _setup_logging() -> None:
    """Surface stage INFO logs (keyframe reduction, extract counts) to stderr."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@app.command()
def process(
    video: Path = typer.Argument(..., help="Path to the source video file."),
    out: Path = typer.Option(
        ..., "--out", "-o", help="Job directory for artifacts (created if absent)."
    ),
    profile: str = typer.Option(
        "generic", "--profile", "-p", help="Domain profile name (profiles/<name>.yaml)."
    ),
    max_cost: float = typer.Option(
        DEFAULT_MAX_COST_USD,
        "--max-cost",
        help="Abort before extract_visuals if estimated LLM spend exceeds this (USD).",
    ),
) -> None:
    """Run the full pipeline over VIDEO, writing artifacts to --out."""
    from videodistill.pipeline import process as run_pipeline

    _setup_logging()
    config = load_config()
    try:
        domain = load_profile(profile)
        results = run_pipeline(video, out, config, domain, max_cost=max_cost)
    except VideoDistillError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"\nPipeline report ({out}, profile={domain.name}):", bold=True)
    for r in results:
        color = {
            "ran": typer.colors.GREEN,
            "stub": typer.colors.YELLOW,
            "skipped": typer.colors.BRIGHT_BLACK,
        }.get(r.status, typer.colors.WHITE)
        marker = {"ran": "✓", "stub": "⚠", "skipped": "·"}.get(r.status, "?")
        typer.secho(f"  {marker} {r.name:<16} {r.status:<8} {r.detail}", fg=color)

    stubs = [r.name for r in results if r.status == "stub"]
    if stubs:
        typer.secho(
            f"\nStopped at stub stage: {stubs[0]}. "
            "Implemented stages ran to completion.",
            fg=typer.colors.YELLOW,
        )


@app.command(name="run-stage")
def run_stage(
    stage: str = typer.Argument(..., help=f"One of: {', '.join(STAGE_ORDER)}"),
    job: Path = typer.Option(
        ..., "--job", "-j", help="Existing job directory to read/write."
    ),
    profile: str = typer.Option(
        "generic", "--profile", "-p", help="Domain profile name (profiles/<name>.yaml)."
    ),
    max_cost: float = typer.Option(
        DEFAULT_MAX_COST_USD,
        "--max-cost",
        help="Cost guard (USD) for extract_visuals; ignored by other stages.",
    ),
) -> None:
    """Re-run a single STAGE against an existing --job directory."""
    _setup_logging()
    config = load_config()
    try:
        domain = load_profile(profile)
        run_single_stage(stage, job, config, domain, max_cost=max_cost)
    except StageNotImplemented as exc:
        typer.secho(
            f"Stage '{stage}' is a stub: {exc}", fg=typer.colors.YELLOW, err=True
        )
        raise typer.Exit(code=2) from exc
    except VideoDistillError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho(
        f"✓ Stage '{stage}' completed. Artifacts in {job}.", fg=typer.colors.GREEN
    )


@app.command(name="eval")
def eval_cmd(
    job: Path = typer.Option(
        ..., "--job", "-j", help="Job directory produced by `process`."
    ),
    profile: str = typer.Option(
        "generic", "--profile", "-p", help="Domain profile name (profiles/<name>.yaml)."
    ),
    grounding_n: int = typer.Option(
        5, "--grounding-n", help="How many notes the grounding judge samples (LLM)."
    ),
) -> None:
    """Evaluate a job directory and write eval_report.md."""
    from videodistill.evals import evaluate

    _setup_logging()
    config = load_config()
    try:
        domain = load_profile(profile)
        report = evaluate(job, domain, config, grounding_n=grounding_n)
    except VideoDistillError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(report)
    typer.secho(f"Written to {job / 'eval_report.md'}", fg=typer.colors.GREEN)


kb_app = typer.Typer(
    help="Knowledge base: ingest notes and search across videos.",
    no_args_is_help=True,
)
app.add_typer(kb_app, name="kb")


def _embedder(config: Config, *, required: bool) -> LLMClient | None:
    if not config.openai_api_key:
        if required:
            raise VideoDistillError(
                "OPENAI_API_KEY is required to embed notes. Set it in .env."
            )
        return None
    from videodistill.llm.cache import CachedLLMClient
    from videodistill.llm.openai_provider import OpenAIProvider

    return CachedLLMClient(
        OpenAIProvider(api_key=config.openai_api_key), config.cache_dir
    )


@kb_app.command("ingest")
def kb_ingest(
    job_dirs: list[Path] = typer.Argument(
        ..., help="One or more job directories containing notes.jsonl."
    ),
    collection: str = typer.Option(..., "--collection", "-c", help="Collection name."),
) -> None:
    """Embed and ingest notes from JOB_DIRS into a KB collection."""
    from videodistill.kb import build_store, ingest_job_dirs

    _setup_logging()
    config = load_config()
    try:
        embedder = _embedder(config, required=True)
        assert embedder is not None
        config.kb_dir.mkdir(parents=True, exist_ok=True)
        store = build_store(config.vector_store, config.kb_dir)
        count = ingest_job_dirs(
            job_dirs, collection, store, embedder, config.embed_model
        )
    except VideoDistillError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho(
        f"✓ Ingested {count} note(s) into collection '{collection}'.",
        fg=typer.colors.GREEN,
    )


@kb_app.command("search")
def kb_search(
    query: str = typer.Argument(..., help="Search query."),
    collection: str = typer.Option(..., "--collection", "-c", help="Collection name."),
    k: int = typer.Option(8, "--k", help="Number of results."),
) -> None:
    """Hybrid-search a KB collection (vector + BM25)."""
    from videodistill.kb import build_store, hybrid_search

    _setup_logging()
    config = load_config()
    embedder = _embedder(config, required=False)
    if embedder is None:
        typer.secho(
            "OPENAI_API_KEY not set — searching with BM25 only (no vectors).",
            fg=typer.colors.YELLOW,
            err=True,
        )
    store = build_store(config.vector_store, config.kb_dir)
    results = hybrid_search(query, store, collection, embedder, config.embed_model, k=k)

    if not results:
        typer.secho("No results.", fg=typer.colors.YELLOW)
        return
    for r in results:
        stamp = (
            f"{int(r.source_timestamp) // 60:02d}:{int(r.source_timestamp) % 60:02d}"
        )
        typer.secho(
            f"[{r.collection}/{r.source_video} @ {stamp}] {r.concept}  "
            f"`{r.canonical_concept_id}`",
            bold=True,
        )
        typer.echo(f"    {r.summary}")


if __name__ == "__main__":
    app()
