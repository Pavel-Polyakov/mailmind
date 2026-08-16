from mailmind.textprep import (
    TRUNCATION_MARKER,
    html_to_text,
    prepare_body,
    strip_quoted,
    truncate,
)


def test_strips_quoted_reply_chain():
    text = "Please confirm.\n\nOn Mon, Aug 10, 2026 at 9:00 AM Bob wrote:\n> old stuff\n> more old"
    assert strip_quoted(text).strip() == "Please confirm."


def test_strips_signature_and_everything_after():
    text = "The order shipped.\n-- \nBob\nSent from wherever"
    assert strip_quoted(text).strip() == "The order shipped."


def test_strips_quote_markers_and_boilerplate():
    text = "Real content\n> quoted line\nUnsubscribe\nView in browser\nMore content"
    assert strip_quoted(text).splitlines() == ["Real content", "More content"]


def test_html_fallback_extracts_text():
    html = "<html><style>b{}</style><body><p>Order&nbsp;#42</p><br><div>Shipped</div></body></html>"
    text = html_to_text(html)
    assert "Order #42" in text
    assert "Shipped" in text
    assert "<" not in text


def test_truncate_marks_the_cut_and_respects_the_cap():
    text = "word " * 500
    result = truncate(text, 100)
    assert result.endswith(TRUNCATION_MARKER)
    assert len(result) <= 100 + len(TRUNCATION_MARKER)


def test_truncate_leaves_short_text_alone():
    assert truncate("short", 100) == "short"


def test_truncate_of_one_long_line_keeps_most_of_the_cap():
    # A boundary far from the cap must not gut the body.
    result = truncate("x" * 500, 100)
    assert len(result) >= 100


def test_prepare_body_prefers_plain_over_html():
    result = prepare_body("plain wins", "<p>html loses</p>", 4000)
    assert result == "plain wins"


def test_prepare_body_falls_back_to_html_when_plain_is_blank():
    assert "html used" in prepare_body("   ", "<p>html used</p>", 4000)


def test_prepare_body_collapses_blank_lines():
    assert prepare_body("a\n\n\n\n\nb", None, 4000) == "a\n\nb"
