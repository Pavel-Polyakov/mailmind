"""Gmail access: read-only OAuth, search, and message parsing.

Gmail stays the source of truth for mail content. Nothing here writes, and the
requested scope makes that a guarantee rather than a promise.
"""

from __future__ import annotations

import base64
import email.utils
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

from .config import CREDENTIALS_FILE, TOKEN_FILE, ensure_config_dir, secure_file
from .db import Email
from .textprep import prepare_body

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailError(Exception):
    pass


def authorize(*, reauth: bool = False, interactive: bool = True):
    """Return usable credentials, running the consent flow when needed."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:  # pragma: no cover - depends on install
        raise GmailError(
            "Google API packages are missing. Install with: "
            "uv add google-api-python-client google-auth-oauthlib"
        ) from exc

    ensure_config_dir()
    if reauth:
        TOKEN_FILE.unlink(missing_ok=True)

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if not interactive:
            raise GmailError("no cached Gmail token; run `mailmind auth` first")
        if not CREDENTIALS_FILE.exists():
            raise GmailError(
                f"no OAuth client secrets at {CREDENTIALS_FILE}.\n"
                "Create a Desktop OAuth client in Google Cloud Console, enable the "
                "Gmail API, and save the downloaded JSON to that path."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
        creds = flow.run_local_server(port=0)

    TOKEN_FILE.write_text(creds.to_json())
    secure_file(TOKEN_FILE)
    return creds


@dataclass
class GmailClient:
    """Thin wrapper over the Gmail API surface this tool needs."""

    service: object
    max_body_chars: int = 4000

    @classmethod
    def connect(cls, *, max_body_chars: int = 4000, interactive: bool = True) -> GmailClient:
        from googleapiclient.discovery import build

        creds = authorize(interactive=interactive)
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return cls(service=service, max_body_chars=max_body_chars)

    def search(self, query: str | None, limit: int | None = None) -> list[str]:
        """Message ids matching a raw Gmail query, newest first."""
        ids: list[str] = []
        page_token = None
        while True:
            batch = min(500, limit - len(ids)) if limit else 500
            request = self.service.users().messages().list(
                userId="me", q=query or "", maxResults=batch, pageToken=page_token
            )
            response = _execute(request)
            ids.extend(m["id"] for m in response.get("messages", []))
            page_token = response.get("nextPageToken")
            if not page_token or (limit and len(ids) >= limit):
                break
        return ids[:limit] if limit else ids

    def fetch(self, message_id: str) -> Email:
        request = self.service.users().messages().get(
            userId="me", id=message_id, format="full"
        )
        return parse_message(_execute(request), self.max_body_chars)


def _execute(request):
    try:
        return request.execute()
    except Exception as exc:  # noqa: BLE001 - surfaced with context by callers
        raise GmailError(str(exc)) from exc


# --- query building -------------------------------------------------------


def build_query(
    query: str | None = None,
    after: str | None = None,
    before: str | None = None,
    since_ms: int | None = None,
) -> str:
    """Merge the raw query with the date flags into one Gmail query.

    `--after` / `--before` mirror Gmail's own operators, which is why they are
    named that way rather than --from/--to: in a tool that also takes a raw
    Gmail query, `--from` reads unavoidably as a sender filter.
    """
    parts: list[str] = []
    if query:
        parts.append(query.strip())
    if after:
        parts.append(f"after:{_as_gmail_date(after)}")
    if before:
        parts.append(f"before:{_as_gmail_date(before)}")
    if since_ms is not None:
        # Gmail accepts unix seconds; +1s so the watermark message is excluded.
        parts.append(f"after:{since_ms // 1000 + 1}")
    return " ".join(parts)


def _as_gmail_date(value: str) -> str:
    value = value.strip()
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y/%m/%d")
    except ValueError as exc:
        raise GmailError(f"dates must be YYYY-MM-DD, got {value!r}") from exc


# --- message parsing ------------------------------------------------------


def parse_message(payload: dict, max_body_chars: int) -> Email:
    """Turn a Gmail `messages.get` response into a stored Email."""
    headers = {
        h.get("name", "").lower(): h.get("value", "")
        for h in payload.get("payload", {}).get("headers", [])
    }
    plain, html_body = _extract_bodies(payload.get("payload", {}))
    internal = payload.get("internalDate")
    internal_ms = int(internal) if internal else None

    return Email(
        message_id=payload["id"],
        thread_id=payload.get("threadId"),
        date=_normalise_date(headers.get("date"), internal_ms),
        internal_date=internal_ms,
        sender=headers.get("from"),
        subject=headers.get("subject"),
        snippet=payload.get("snippet"),
        body=prepare_body(plain, html_body, max_body_chars),
        labels=list(payload.get("labelIds", [])),
        list_unsubscribe=headers.get("list-unsubscribe"),
    )


def _extract_bodies(part: dict) -> tuple[str | None, str | None]:
    """Walk the MIME tree, collecting the first text/plain and text/html."""
    plain: str | None = None
    html_body: str | None = None
    for node in _walk(part):
        mime = node.get("mimeType", "")
        data = node.get("body", {}).get("data")
        if not data:
            continue
        if mime == "text/plain" and plain is None:
            plain = _decode(data)
        elif mime == "text/html" and html_body is None:
            html_body = _decode(data)
    return plain, html_body


def _walk(part: dict) -> Iterator[dict]:
    yield part
    for child in part.get("parts", []) or []:
        yield from _walk(child)


def _decode(data: str) -> str:
    raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    return raw.decode("utf-8", errors="replace")


def _normalise_date(header: str | None, internal_ms: int | None) -> str | None:
    if header:
        try:
            return email.utils.parsedate_to_datetime(header).isoformat()
        except (TypeError, ValueError):
            pass
    if internal_ms:
        return datetime.fromtimestamp(internal_ms / 1000, tz=timezone.utc).isoformat()
    return None
