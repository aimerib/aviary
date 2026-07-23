"""Filters for third-party RP data. The minor-content rules are safety-critical:
a regression here silently admits the exact content the filter exists to exclude,
so each signal is pinned individually, as is the false positive we rejected.

Pure functions only — no network, no parquet, runs under the socket ban.
"""

from __future__ import annotations

from aviary.external.rp_reasoning import classify, substitute_placeholders


def conv(*turns: tuple[str, str]) -> list[dict]:
    return [{"from": frm, "value": val} for frm, val in turns]


def ok_row(text: str = "The tavern is loud.") -> list[dict]:
    """A row that survives every filter, so a test can isolate one signal."""
    think = "<think>\nSCENE: tavern\nCHARACTERS: Mara\nPLAN: greet\n</think>\n"
    return conv(
        ("system", "You are Mara, a tired innkeeper."),
        ("human", "I walk in."),
        ("gpt", think + text),
    )


def test_clean_row_survives():
    assert classify(ok_row()) is None


# --- minor-coded content: each signal pinned individually --------------------


def test_minor_numeric_age():
    row = ok_row("She is a 15-year-old girl.")
    assert classify(row) == "minor_coded"


def test_minor_word_age():
    assert classify(ok_row("a fifteen-year-old boy")) == "minor_coded"


def test_minor_age_field():
    # character-card convention: "Age: 13"
    assert classify(ok_row("Name: Kai\nAge: 13\n")) == "minor_coded"


def test_minor_keywords():
    for kw in ("lolicon", "shota", "underage", "preteen", "jailbait"):
        assert classify(ok_row(f"tagged {kw} content")) == "minor_coded", kw


def test_minor_child_body():
    assert classify(ok_row("described her child's body")) == "minor_coded"


def test_bare_word_minor_is_not_a_signal():
    # Verified false positive: 'minor' matched "a minor detail" and only 40% of
    # its hits co-occurred with sexual content. Excluding it is deliberate.
    assert classify(ok_row("That is a minor detail, honestly.")) is None


def test_adult_age_is_not_a_signal():
    assert classify(ok_row("She is a 25-year-old woman.")) is None


def test_minor_dropped_even_without_sexual_content():
    # The sexual-marker regex is a crude proxy; the age signal alone must drop.
    assert classify(ok_row("The 12-year-old ate breakfast quietly.")) == "minor_coded"


def test_school_setting_only_under_strict():
    row = ok_row("They met behind the middle school.")
    assert classify(row) is None
    assert classify(row, strict_minor=True) == "minor_school_setting"


# --- provenance / structure --------------------------------------------------


def test_claude_rows_only_dropped_when_requested():
    row = ok_row("The assistant is a derivative of Claude named CLAUD3.")
    assert classify(row) is None
    assert classify(row, drop_claude=True) == "claude_derived"


def test_leaked_meta_preamble():
    assert classify(ok_row("I have read the <rules> and know what I must do.")) == "leaked_meta"


def test_reasoning_must_lead_every_assistant_turn():
    think = "<think>\nSCENE: x\nCHARACTERS: y\nPLAN: z\n</think>\n"
    partial = conv(
        ("system", "s"),
        ("human", "a"),
        ("gpt", think + "first"),
        ("human", "b"),
        ("gpt", "second, with no reasoning"),
    )
    assert classify(partial) == "think_incomplete"


def test_reasoning_must_lead_not_merely_appear():
    trailing = conv(
        ("system", "s"),
        ("human", "a"),
        ("gpt", "prose first\n<think>\nSCENE: x\nCHARACTERS: y\nPLAN: z\n</think>"),
    )
    assert classify(trailing) == "think_incomplete"


def test_consecutive_assistant_turns_rejected():
    think = "<think>\nSCENE: x\nCHARACTERS: y\nPLAN: z\n</think>\n"
    row = conv(("system", "s"), ("human", "a"), ("gpt", think + "one"), ("gpt", think + "two"))
    assert classify(row) == "consecutive_assistant_turns"


def test_offschema_think_rejected():
    row = conv(("system", "s"), ("human", "a"), ("gpt", "<think>\njust musing\n</think>\nhi"))
    assert classify(row) == "think_offschema"


def test_empty_turn_rejected():
    think = "<think>\nSCENE: x\nCHARACTERS: y\nPLAN: z\n</think>\n"
    row = conv(("system", "s"), ("human", "   "), ("gpt", think + "hi"))
    assert classify(row) == "empty_turn"


# --- placeholder handling ----------------------------------------------------


def test_user_placeholder_substituted_not_dropped():
    row = conv(("system", "You talk to {{user}}."), ("human", "hi"), ("gpt", "Hello {{user}}!"))
    out = substitute_placeholders(row, "abc123")
    assert out is not None
    assert "{{user}}" not in out[0]["value"]
    assert "{{user}}" not in out[2]["value"]
    # the same name is used consistently across the whole conversation
    name = out[2]["value"].removeprefix("Hello ").removesuffix("!")
    assert name and name in out[0]["value"]


def test_substitution_is_stable_for_a_row_id():
    row = conv(("gpt", "Hi {{user}}"))
    assert substitute_placeholders(row, "row-1") == substitute_placeholders(row, "row-1")


def test_unresolvable_placeholder_rejects_row():
    # {{char}} / {{random}} cannot be resolved without guessing, so the row goes.
    assert substitute_placeholders(conv(("gpt", "Hi {{char}}")), "x") is None
    assert substitute_placeholders(conv(("gpt", "roll {{random}}")), "x") is None
