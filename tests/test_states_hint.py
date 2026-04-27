"""Verify that author-declared states get injected into planner prompts."""

from opencanvas.agents import _states_hint


def test_no_hint_for_empty_declaration():
    assert _states_hint([], "appearance state names") == ""


def test_no_hint_when_only_default():
    assert _states_hint(["default"], "state names") == ""


def test_hint_lists_meaningful_states():
    s = _states_hint(["intact", "burned", "default"], "state names")
    assert "Prefer one of these state names when applicable" in s
    assert "intact" in s
    assert "burned" in s
    # Default is filtered out — it's a sentinel, not a real state
    assert "['intact', 'burned']" in s


def test_hint_label_passed_through():
    s = _states_hint(["tuxedo", "raincoat"], "appearance state names")
    assert "appearance state names" in s
