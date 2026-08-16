# Mailmind — Implementation Spec

## Goal

Build a personal CLI tool that scans Gmail, classifies emails with an LLM, stores the results in SQLite, and allows later refinement of the discovered taxonomy.

The tool is analysis-only for now. It must not modify Gmail.

## Tech

* Python 3.14
* `uv`
* SQLite
* Gmail API
* Pluggable LLM provider

Use reasonable implementation choices for the rest.

## CLI

Provide:

```bash
mailmind auth
mailmind scan
mailmind refine
mailmind report
mailmind diff
mailmind runs
```

### Configuration

Flags override config file, which overrides defaults.

Config lives at `~/.config/mailmind/config.toml` and may set at least:

```text
db
model
concurrency
max_body_chars
```

API keys come from the environment (`OPENAI_API_KEY` and equivalents), never from the config file.

Create the database file and the config directory with mode `0600`.

## `mailmind auth`

Perform the Gmail OAuth flow and cache the token.

```text
--reauth        discard the cached token and re-consent
```

Client secrets are read from `~/.config/mailmind/credentials.json`.
The token is cached at `~/.config/mailmind/token.json`, mode `0600`.

`scan` triggers this flow automatically on first use.

## `mailmind scan`

Fetch emails from Gmail and classify them with an LLM.

Support:

```text
--after DATE
--before DATE
--since-last
--query GMAIL_QUERY
--limit N
--model PROVIDER:MODEL
--base-url URL
--db PATH
--stdout
--concurrency N
--max-body-chars N
--no-store-body
--reuse-labels
--new-run
--resume RUN_ID
--dry-run
```

`--query` accepts a raw Gmail search query. `--after` / `--before` mirror Gmail's own `after:` / `before:` operators, and are merged into the effective query.

`--since-last` scans forward from the high-water mark recorded for the same query, and is mutually exclusive with `--after`.

`--dry-run` fetches and prepares prompts, reports the email count and estimated token/cost, and makes no LLM calls.

Example:

```bash
mailmind scan \
  --after 2026-08-01 \
  --before 2026-08-16 \
  --limit 200 \
  --model openai:<model> \
  --db mail.db
```

### Gmail

Use read-only OAuth permissions.

Classification granularity is **one message, not one thread**. Thread id is stored so thread-level analysis remains possible later.

For each message, retrieve enough data for useful classification, including at least:

```text
gmail message id
thread id
date
sender
subject
snippet
plain-text body where available
Gmail labels
List-Unsubscribe where available
```

Gmail remains the source of truth. Do not store raw MIME unless needed.

### Body preparation

Before prompting, strip quoted reply chains, signatures, and boilerplate, then truncate to `--max-body-chars` (default `4000`).

The effective cap is recorded on the run. Two runs are only comparable if their caps match, so this must not be an implicit constant.

## Classification

Each email produces structured LLM output.

Request it through the provider's JSON-schema or tool-use mode and validate it against the schema below. On validation failure, retry once with the validation error fed back; if it fails again, persist an error row and continue.

```json
{
  "summary": "Flight BA281 departure changed from 12:30 to 14:10.",
  "type": "flight_schedule_change",
  "category": "Travel",
  "importance": "high",
  "needs_action": true,
  "suggested_action": "inbox",
  "confidence": 0.94,
  "reasoning_short": "Flight time changed and may require acknowledgement."
}
```

All fields are **required and non-null**. Providers omit whatever is not marked required.

### `summary`

One short factual sentence describing the important content of the email.

Preserve useful concrete information where present, such as names, dates, times, amounts, deadlines, booking/order identifiers, and status changes.

Avoid generic filler.

### `type`

Describes what kind of email this is.

Types are **not predefined**. The LLM may discover new types.

Examples:

```text
receipt
delivery_notification
flight_schedule_change
school_announcement
login_code
newsletter
promotion
```

Free-form labels on independent calls will produce many near-duplicates. That is expected and is what `refine` exists to clean up.

`--reuse-labels` optionally feeds the labels already seen in the current run into the prompt as preferred candidates. This reduces drift but makes results order-dependent, which weakens reproducibility and constrains parallelism. It is **off by default** and recorded on the run.

### `category`

Describes the broader context or area of life.

Categories are also **not predefined**.

Examples:

```text
Travel
School
Finance
Shopping
Work
Car
Personal
```

### Other fields

`importance` should use a small stable set such as:

```text
critical
high
normal
low
```

`needs_action` is boolean.

`suggested_action` should use a small stable set:

```text
inbox
archive
delete
unsubscribe
```

The suggestion is advisory only.

`confidence` is a value from 0 to 1. Self-reported confidence is not calibrated; treat it as a rough sort key, not a probability. Any low-confidence threshold in `report` is a flag, not a constant.

`reasoning_short` should be very brief and mainly useful for debugging classification quality.

## LLM providers

The implementation must not be tightly coupled to one LLM provider.

Models are selected as `provider:model`. Initially support an API-based provider plus an OpenAI-compatible provider driven by `--base-url`, which covers Ollama, vLLM, and similar local endpoints.

The selected provider/model should be configurable from the CLI.

### Throughput and failure handling

```text
bounded concurrency, default 8, set by --concurrency
retry with exponential backoff on 429 and 5xx
per-email error rows recording status, error, and attempt count
```

Failures are recorded, not lost. A rerun retries only the failed and unprocessed emails.

## SQLite

SQLite is a reusable canonical artifact, not just temporary output.

Persist:

* fetched email data;
* classification runs;
* classification results;
* refinement runs and their label mappings.

Keep source email data separate from derived LLM results.

Do not overwrite previous classification results when rerunning the same emails with:

* another model;
* another prompt;
* another classifier version.

Multiple classification runs over the same emails must be possible.

Raw structured LLM output is retained on success as well as on failure, for debugging.

Email bodies are stored once, keyed by gmail message id, and referenced by every run. Bodies are never duplicated per run.

`--no-store-body` keeps only subject, snippet, and derived summary. The database will otherwise contain plaintext login codes and account details.

### Table sketch

```text
email               (gmail message id, thread id, date, sender, subject,
                     snippet, body, labels, list_unsubscribe)
classification_run  (id, fingerprint, status, model, base_url,
                     prompt_version, prompt_text, schema_version,
                     query, max_body_chars, reuse_labels, created_at)
classification      (run_id, message_id, summary, type, category,
                     importance, needs_action, suggested_action,
                     confidence, reasoning_short, raw_response,
                     status, error, attempts)
refine_run          (id, source_run_id, kind, model, prompt_version,
                     prompt_text, created_at)
label_map           (refine_run_id, kind, source_label, canonical_label)
scan_watermark      (query, last_scanned_at)
```

`status` on `classification_run` is one of:

```text
in_progress
completed
failed
```

### Run identity and resumption

`fingerprint` is a hash of the model, prompt version, schema version, and the effective scan parameters (query, limit, body cap, reuse-labels).

Resolution order for `mailmind scan`:

```text
--resume RUN_ID   resume that run explicitly
--new-run         always start a new run
otherwise         resume the most recent in_progress run with a matching
                  fingerprint, else start a new run
```

A resumed run skips emails already classified successfully and retries the rest. This is what makes "safe to interrupt" and "do not re-call the LLM unnecessarily" compatible rather than contradictory.

### Prompt versioning

`prompt_version` is **derived, not hand-maintained**: `sha256(rendered prompt template + JSON schema)[:12]`. The full prompt text is stored alongside it on the run.

A version that must be remembered to be bumped is hidden state, and reproducibility claims stop holding.

## `mailmind refine`

Refine the taxonomy produced by a classification run.

Example:

```bash
mailmind refine --run 8 --kind type --model openai:<model>
```

```text
--run RUN_ID
--kind type|category|both     default: both
--model PROVIDER:MODEL
```

The command inspects the discovered labels, their frequencies, and representative examples.

Its purpose is to merge semantically equivalent or unnecessarily specific labels into a smaller coherent taxonomy.

For example:

```text
online_purchase_receipt
purchase_confirmation
order_receipt
transaction_receipt
```

may be normalized to:

```text
receipt
```

`type` and `category` are refined independently, with separate prompts and separate rows discriminated by `kind`.

### Output model

A refine run produces a **label mapping**, not new classifications:

```text
label_map(refine_run_id, kind, source_label, canonical_label)
```

Reports apply the mapping as a left join at query time. Labels absent from the mapping pass through unchanged, so a refine run stays usable after a later scan introduces new labels.

The source classifications must remain unchanged.

Each refine run is persisted separately so that refinement prompts/models can be experimented with and compared.

Do not hardcode the final taxonomy.

## `mailmind runs`

List classification and refine runs: id, kind, model, prompt version, email count, status, timestamp.

Every other command takes run ids as input, so they must be discoverable.

## `mailmind report`

Provide simple human-readable inspection of the stored dataset.

```text
--run RUN_ID
--refine REFINE_RUN_ID
--format table|json|csv       default: table
--min-confidence FLOAT
--type LABEL
--category LABEL
--limit N
```

At minimum support useful views such as:

```text
emails by type
emails by category
emails by sender/domain
emails by suggested action
emails requiring action
low-confidence classifications
```

Support drilling into the underlying emails, for example:

```bash
mailmind report emails --run 8 --type receipt --limit 50
```

It should be possible to inspect both raw classification runs and refined taxonomy runs. Passing `--refine` applies that mapping to the view.

`json` and `csv` output exist because the database is meant to feed the later rule-proposal stage. Text tables alone are a dead end.

## `mailmind diff`

Compare two classification runs over their overlapping emails.

```bash
mailmind diff --run 8 --run 11
```

Report agreement rate per field (`type`, `category`, `importance`, `suggested_action`, `needs_action`) and the most common disagreements with example emails.

This is how a model or prompt change is actually evaluated. The run-immutability rules exist to make it possible.

## stdout

`scan` should be usable without SQLite for quick experiments and should print useful classification output to stdout.

When a database is supplied, results should also be persisted.

Progress/errors should be understandable during long runs.

## Robustness

The scanner should be safe to interrupt and rerun; resumption semantics are defined above.

Handle individual malformed emails or LLM failures without losing the entire run.

Provide useful error messages.

## Scope constraints

This version must **not modify Gmail**.

Do not implement:

```text
archive
delete
label creation/modification
Gmail filters
unsubscribe
retention policies
snooze
PWA/web UI
```

Do not generate proposed Gmail filters during per-email classification.

A later stage/tool will use the collected and refined dataset to propose Gmail cleanup rules and filters.

Likely future pipeline:

```text
scan
  ↓
refine
  ↓
report
  ↓
propose rules
  ↓
review
  ↓
apply to Gmail
```

The current implementation should stop at `report`.

## Design principles

* Optimize for a personal tool, not a multi-user service.
* Keep the implementation small and understandable.
* Gmail is the source of truth for email content.
* SQLite is the source of truth for analysis results.
* Preserve intermediate results instead of mutating them.
* Prefer reproducible runs over hidden state.
* Derive versions and identity; do not rely on remembering to bump them.
* LLM classifications are opinions, not facts.
* Record failures rather than dropping them.
* Keep analysis strictly separate from future Gmail mutations.
* Do not over-engineer for hypothetical future requirements.
