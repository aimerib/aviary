from __future__ import annotations

from aviary.gates.spans import extract_protected_spans, mask_spans, restore_spans

SAMPLE = (
    "I saved it to /home/user/notes/haiku.txt after running `wc -l` — "
    'the file says "rain taps the window" and is 62 bytes, about 3.5% of quota. '
    "See https://example.com/docs for the write_file docs.\n"
    "```python\nprint('hi')\n```"
)


def test_extracts_expected_kinds():
    kinds = {s.kind for s in extract_protected_spans(SAMPLE)}
    assert {"abs_path", "inline_code", "quoted", "number", "url", "fenced_code"} <= kinds
    ident = [s for s in extract_protected_spans(SAMPLE) if s.kind == "identifier"]
    assert any(s.text == "write_file" for s in ident)


def test_mask_restore_roundtrip_identity():
    spans = extract_protected_spans(SAMPLE)
    masked, mapping = mask_spans(SAMPLE, spans)
    for span in spans:
        assert span.text not in masked or span.kind == "number"  # numbers may repeat
    assert restore_spans(masked, mapping) == SAMPLE


def test_restore_survives_prose_rewrite():
    text = "The result lives in /tmp/out.log and took 42 seconds."
    spans = extract_protected_spans(text)
    masked, mapping = mask_spans(text, spans)
    rewritten = masked.replace("The result lives in", "Your output landed in").replace(
        "and took", "after a glacial"
    )
    restored = restore_spans(rewritten, mapping)
    assert restored is not None
    assert "/tmp/out.log" in restored and "42" in restored


def test_restore_fails_on_dropped_placeholder():
    text = "Wrote /tmp/a.txt and /tmp/b.txt today."
    masked, mapping = mask_spans(text, extract_protected_spans(text))
    assert len(mapping) == 2
    dropped = masked.replace("⟦S1⟧", "")
    assert restore_spans(dropped, mapping) is None


def test_restore_fails_on_duplicated_placeholder():
    text = "The file is /etc/hosts right now."
    masked, mapping = mask_spans(text, extract_protected_spans(text))
    assert restore_spans(masked + " and also ⟦S0⟧", mapping) is None


def test_restore_fails_on_invented_placeholder():
    text = "Check /var/log/sys.log please."
    masked, mapping = mask_spans(text, extract_protected_spans(text))
    assert restore_spans(masked + " ⟦S9⟧", mapping) is None


def test_overlapping_spans_resolve_by_priority():
    text = 'run `curl https://api.example.com/v1` with "quotes" inside'
    spans = extract_protected_spans(text)
    starts = [s.start for s in spans]
    assert starts == sorted(starts)
    for a, b in zip(spans, spans[1:], strict=False):
        assert a.end <= b.start  # no overlaps survive
