"""Prompt templates.

The template text is stored verbatim on every run, and its hash is the prompt
version. Editing anything in this file therefore produces a new version
automatically -- there is no counter to remember to bump.
"""

from __future__ import annotations

CLASSIFY_SYSTEM = """\
You classify a single email for a personal mail-triage tool.

Return one JSON object matching the provided schema. Rules:

- summary: one short factual sentence. Keep concrete details that make the mail
  actionable: names, dates, times, amounts, deadlines, booking or order
  identifiers, status changes. No filler like "this email informs you that".
- type: what kind of email this is, snake_case, e.g. receipt,
  delivery_notification, flight_schedule_change, login_code, newsletter.
  Types are not predefined; use the most natural specific label.
- category: the broader area of life, capitalised, e.g. Travel, School,
  Finance, Shopping, Work, Car, Personal. Also not predefined.
- importance: critical, high, normal, or low, judged for the recipient.
- needs_action: true only if the recipient must actually do something.
- suggested_action: inbox, archive, delete, or unsubscribe. Advisory only.
- confidence: 0 to 1, your own certainty about type and category.
- reasoning_short: a handful of words, for debugging classification quality.
"""

REUSE_LABELS_HINT = """\

Labels already used in this run are listed with each email. Prefer an existing
label when one genuinely fits; invent a new one when none does.
"""

CLASSIFY_USER = """\
From: {sender}
Date: {date}
Subject: {subject}
Gmail labels: {labels}
List-Unsubscribe: {list_unsubscribe}

{body}
"""

REFINE_SYSTEM = """\
You are consolidating a taxonomy of {kind} labels discovered by an email
classifier. The labels were invented independently per email, so the list
contains near-duplicates and labels that are more specific than useful.

Return one JSON object matching the provided schema: a list of mappings from
source_label to canonical_label. Rules:

- Merge labels that mean the same thing, e.g. online_purchase_receipt,
  order_receipt, and transaction_receipt all map to receipt.
- Merge labels that are needlessly specific into their natural parent, but keep
  distinctions that would change how someone triages the mail.
- Prefer an existing label as the canonical form; invent a name only when no
  member of the group is a good representative.
- Keep the style of the input: {style}.
- Include every source label exactly once, including labels that map to
  themselves.
"""

REFINE_USER = """\
Labels with frequencies and example summaries:

{labels}
"""

STYLE = {"type": "snake_case", "category": "Capitalised words"}
