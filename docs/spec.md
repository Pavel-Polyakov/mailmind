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
mailmind scan
mailmind refine
mailmind report
```

## `mailmind scan`

Fetch emails from Gmail and classify them with an LLM.

Support:

```text
--from DATE
--to DATE
--query GMAIL_QUERY
--limit N
--model MODEL
--db PATH
--stdout
```

`--query` should accept a raw Gmail search query.

Example:

```bash
mailmind scan \
  --from 2026-08-01 \
  --to 2026-08-16 \
  --limit 200 \
  --model openai:<model> \
  --db mail.db
```

### Gmail

Use read-only OAuth permissions.

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

## Classification

Each email should produce structured LLM output:

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

`confidence` is a value from 0 to 1.

`reasoning_short` should be very brief and mainly useful for debugging classification quality.

## LLM providers

The implementation must not be tightly coupled to one LLM provider.

Initially support an API-based provider.

It should be straightforward to later point the tool at a local model through an OpenAI-compatible endpoint, Ollama, vLLM, or similar.

The selected provider/model should be configurable from the CLI.

## SQLite

SQLite is a reusable canonical artifact, not just temporary output.

Persist:

* fetched email data;
* classification runs;
* classification results;
* refinement results.

Keep source email data separate from derived LLM results.

Do not overwrite previous classification results when rerunning the same emails with:

* another model;
* another prompt;
* another classifier version.

Multiple classification runs over the same emails must be possible.

Store enough metadata to identify how a classification was produced, including model and prompt/classifier version.

Raw structured LLM output should also be retained for debugging.

Avoid unnecessary duplication of email bodies between runs.

## `mailmind refine`

Refine the taxonomy produced by a classification run.

Example:

```bash
mailmind refine --run 8 --model openai:<model>
```

The command should inspect the discovered types and categories, their frequencies, and representative examples.

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

The source classifications must remain unchanged.

Each refine run must be persisted separately so that refinement prompts/models can be experimented with and compared.

Do not hardcode the final taxonomy.

## `mailmind report`

Provide simple human-readable inspection of the stored dataset.

At minimum support useful views such as:

```text
emails by type
emails by category
emails by sender/domain
emails by suggested action
emails requiring action
low-confidence classifications
```

It should be possible to inspect both raw classification runs and refined taxonomy runs.

Output can be plain terminal tables/text.

## stdout

`scan` should be usable without SQLite for quick experiments and should print useful classification output to stdout.

When a database is supplied, results should also be persisted.

Progress/errors should be understandable during long runs.

## Robustness

The scanner should be safe to interrupt and rerun.

Avoid unnecessarily calling the LLM again for work already persisted unless the user explicitly starts a new classification run.

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
* LLM classifications are opinions, not facts.
* Keep analysis strictly separate from future Gmail mutations.
* Do not over-engineer for hypothetical future requirements.
