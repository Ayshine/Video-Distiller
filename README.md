# Video-Distiller

Turn any informational video — lectures, talks, screencasts, whole courses —
into structured, timestamped notes and **your own local vector database**, ready
for retrieval-augmented search and Q&A over an entire collection.

Three models do the reading. A **vision-language model (VLM)** captures what's on
screen (slides, code, diagrams, whiteboard); **ASR** transcribes what's said; an
**LLM** fuses the two — aligned by timestamp — into grounded, concept-level notes
you can review in roughly a tenth of the watch time. Those notes are then embedded
into an on-disk [Qdrant](https://qdrant.tech) index with **hybrid retrieval**
(dense vectors + BM25), so a video collection becomes a queryable knowledge base.

Model access is fully abstracted (OpenAI / Gemini VLM / any provider), and all
domain knowledge lives **only** in `profiles/*.yaml`, never in pipeline code —
point it at a different profile and it distills a different subject.

The repo covers two layers: **distill** (video → timestamped notes) and
**knowledge base** (notes → a local vector DB). Downstream retrieval — cross-video
synthesis, RAG Q&A, spaced-repetition quizzes — is a separate project that
consumes the vector DB this one produces.

## Layer 1 — distill pipeline

```
ingest → transcribe → detect_scenes → extract_visuals → align → distill → render
```

Each stage reads and writes typed [Pydantic](https://docs.pydantic.dev)
artifacts (JSON) in a job directory, so any stage can be re-run in isolation.
Stages never import one another and never hardcode domain terms — they take a
`DomainProfile` and read behaviour from it.

| Stage             | Output artifact                 |
| ----------------- | ------------------------------- |
| `ingest`          | `video_meta.json` + `audio.wav` |
| `transcribe`      | `transcript.json`               |
| `detect_scenes`   | `keyframes.json`                |
| `extract_visuals` | `visuals.json`                  |
| `align`           | `aligned.json`                  |
| `distill`         | `notes.json`                    |
| `render`          | `digest.md`, `notes.jsonl`      |

An `eval` command grades a finished job (code-compile rate, vocabulary
hit-rate, LLM grounding spot-check, cost/compression/timing).

## Layer 2 — knowledge base (the vector DB)

Ingest many videos' notes into an embedded local [Qdrant](https://qdrant.tech)
(no Docker) and search them with **hybrid retrieval** — dense vectors for
meaning plus BM25 for exact tokens (a literal identifier or symbol stays
findable), fused with reciprocal rank fusion.

This is the final stage of the pipeline: the resulting vector DB is the artifact
a downstream retrieval / Q&A application consumes.

## Setup

Requires **Python 3.11+**, [uv](https://docs.astral.sh/uv/), and **ffmpeg**.

```bash
brew install ffmpeg          # macOS
uv sync --extra dev          # install project + dev tools
cp .env.example .env         # add OPENAI_API_KEY (used by the LLM stages)
uv run pre-commit install    # optional git hooks
```

## Usage

Distill videos into notes (`<...>` are values you supply):

```bash
# Full pipeline on one video (default profile: generic)
uv run videodistill process <video.mp4> --out jobs/<job>

# ...or with a domain profile you added under profiles/
uv run videodistill process <video.mp4> --out jobs/<job> --profile <profile>

# Re-run a single stage against an existing job directory
uv run videodistill run-stage transcribe --job jobs/<job>

# Grade a finished job
uv run videodistill eval --job jobs/<job> --profile <profile>
```

### Build your own vector DB

Distill each video once (above), then embed all of the resulting notes into a
single named **collection** and search across them. The collection is an on-disk
[Qdrant](https://qdrant.tech) index — that *is* your vector database, no server
to run.

```bash
# 1. Embed the notes from many job dirs into one collection
uv run videodistill kb ingest jobs/<job1> jobs/<job2> --collection <collection>

# 2. Hybrid (vector + BM25) search over everything you ingested
uv run videodistill kb search "<query>" --collection <collection> --k 8
```

Re-run `kb ingest` with more job dirs to grow the same collection. The stored
vectors are what a downstream retrieval / Q&A app would query.

## Profiles

A profile is the only place domain knowledge lives. The `generic` profile ships
by default and makes no assumptions (empty vocabulary, no code verification).

Add a domain by dropping a `profiles/<name>.yaml` file — no code changes. A
profile can seed the ASR prompt with domain vocabulary, declare which code
languages to expect, and (optionally) define a verification command used by the
eval harness. Schema (see [`videodistill/profile.py`](videodistill/profile.py)):

```yaml
name: <str>
description: <str>
vocabulary: [<term>, ...]        # seeds ASR prompt + vocabulary eval
code_languages: [<lang>, ...]
concept_id_prefix: <str>
verification:                    # nullable
  kind: compile_check
  command: "<syntax-check command> {file}"   # {file} = a snippet on disk
distill_hints: <str>             # appended to the distill prompt
```

## Configuration

Env vars with local defaults (see [`videodistill/config.py`](videodistill/config.py)):
`OPENAI_API_KEY`, `VIDEODISTILL_ASR_MODEL` (`small`), `VIDEODISTILL_VISION_MODEL`
(`gpt-4o`), `VIDEODISTILL_DISTILL_MODEL` (`gpt-4o-mini`), `VIDEODISTILL_EMBED_MODEL`
(`text-embedding-3-small`), `VIDEODISTILL_CACHE_DIR` (`.cache`). LLM responses are
cached by content hash in `.cache/`, so re-running a stage during development is
free.

**Vision backend.** The `extract_visuals` stage can be served by either OpenAI or
Google Gemini, selected with `VIDEODISTILL_VISION_PROVIDER` (`openai` | `gemini`).
Setting it to `gemini` (add `GEMINI_API_KEY`, thinking disabled) trades a small
amount of quality for a large drop in per-frame cost — useful when distilling long
courses. The rest of the pipeline is unchanged; the provider abstraction routes
only the vision call.

**Vector store.** `VIDEODISTILL_VECTOR_STORE` (default `qdrant`) picks the KB
backend. Qdrant runs embedded/on-disk with no server; other backends (pgvector,
Chroma) can be added behind the `VectorStore` protocol — see Design notes.

## Development

```bash
uv run pytest              # tests (generate a synthetic fixture on first run)
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy videodistill   # strict type check
```

## Design notes

- **Provider abstraction.** All model access (`complete`, `vision`, `embed`)
  goes through [`videodistill/llm/`](videodistill/llm/). Pipeline code never
  imports an SDK directly, so the backend can be swapped
  (OpenAI → Anthropic → Bedrock) without touching stage logic.
- **Pluggable vector store.** The KB talks to the DB only through the small
  `VectorStore` protocol in [`videodistill/kb/store.py`](videodistill/kb/store.py)
  (four methods over neutral `VectorPoint` / `StoredPoint` values). The shipped
  backend is embedded Qdrant; adding **pgvector**, **Chroma**, etc. is a matter
  of implementing the protocol and registering it in `build_store` — ingest and
  search import no specific backend.
- **AWS-portable.** No global state; stages take explicit input/output paths and
  read config from the environment — a straight path to S3 + Step Functions +
  Fargate later.
