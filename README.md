# mailmind

Scan Gmail, classify each email with an LLM, and keep the results in SQLite so
different models and prompts can be compared later.

The tool is analysis-only. It reads Gmail with a read-only scope and never
modifies your mailbox.

See [docs/spec.md](docs/spec.md) for the full specification.

## Install

```bash
uv sync
```

## Setup

1. Create a **Desktop** OAuth client in the Google Cloud Console and enable the
   Gmail API.
2. Save the downloaded JSON as `~/.config/mailmind/credentials.json`.
3. Authorise:

```bash
uv run mailmind auth
```

Export a key for whichever provider you use:

```bash
export OPENAI_API_KEY=...
```

Optional defaults in `~/.config/mailmind/config.toml`:

```toml
db = "~/mail.db"
model = "openai:gpt-4o-mini"
concurrency = 8
max_body_chars = 4000
```

## Use

```bash
# What will this cost?
uv run mailmind scan --after 2026-08-01 --limit 200 --dry-run

# Classify into a database
uv run mailmind scan --after 2026-08-01 --limit 200 --db mail.db

# Quick experiment, no database, results printed
uv run mailmind scan --limit 10 --stdout

# Everything new since the last scan of this query
uv run mailmind scan --query "in:inbox" --since-last --db mail.db

# Consolidate the discovered taxonomy
uv run mailmind runs --db mail.db
uv run mailmind refine --run 1 --db mail.db

# Look at the results
uv run mailmind report type --run 1 --refine 1 --db mail.db
uv run mailmind report needs-action --run 1 --db mail.db
uv run mailmind report emails --run 1 --type receipt --limit 50 --db mail.db
uv run mailmind report type --run 1 --format json --db mail.db

# Did the new model actually classify differently?
uv run mailmind scan --after 2026-08-01 --limit 200 --model openai:gpt-4o --db mail.db
uv run mailmind diff --run 1 --run 2 --db mail.db
```

## How runs work

Every scan is an immutable run. Rerunning the same emails with a different
model, prompt, or body cap creates a new run instead of overwriting the old
one, which is what makes `mailmind diff` meaningful.

A run's identity is a fingerprint over the model, prompt version, and scan
parameters. Interrupt a scan and rerun the same command and it continues where
it stopped; pass `--new-run` to force a fresh one. Emails that failed are stored
as error rows, so a rerun retries exactly those.

`refine` never rewrites classifications. It stores a mapping from discovered
labels to canonical ones, and `report --refine N` applies that mapping at query
time. Labels the mapping has not seen pass through unchanged.

## Local models

```bash
uv run mailmind scan --model ollama:llama3.1 --limit 10 --stdout
uv run mailmind scan --model openai-compatible:my-model \
  --base-url http://localhost:8000/v1 --limit 10 --stdout
```

## Privacy

The database holds plaintext mail, including login codes, so it is created
`0600`. `--no-store-body` keeps only subject, snippet, and the derived summary;
classification then works from the snippet alone, for that run and any resume of
it.

## Tests

```bash
uv run pytest
```
