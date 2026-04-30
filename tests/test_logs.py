"""Logger writes JSONL + per-shot decisions to out_dir/logs/."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from opencanvas.logs import Logger, _stringify_prompt


class _FakeOutput(BaseModel):
    foo: int
    bar: str


def test_log_call_appends_jsonl(tmp_path: Path):
    log = Logger(tmp_path)
    log.log_call(
        label="continuation_decision",
        instructions="be a planner",
        prompt="prev: x\ncurr: y",
        output=_FakeOutput(foo=1, bar="hi"),
        output_type="ContinuationDecision",
        latency_ms=42,
        model="gemma4:31b",
    )
    log.log_call(
        label="continuation_decision",
        instructions="be a planner",
        prompt="prev: a\ncurr: b",
        output=_FakeOutput(foo=2, bar="bye"),
        output_type="ContinuationDecision",
        latency_ms=44,
        model="gemma4:31b",
    )

    f = tmp_path / "logs" / "llm" / "continuation_decision.jsonl"
    assert f.exists()
    lines = f.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["instructions"] == "be a planner"
    assert first["output"] == {"foo": 1, "bar": "hi"}
    assert first["latency_ms"] == 42
    assert first["model"] == "gemma4:31b"


def test_log_decision_writes_per_shot_json(tmp_path: Path):
    log = Logger(tmp_path)
    log.log_decision(4, {"shot_index": 4, "continuation_mode": "previous_frame_continuation"})
    f = tmp_path / "logs" / "decisions" / "shot_0004.json"
    assert f.exists()
    data = json.loads(f.read_text())
    assert data["shot_index"] == 4


def test_log_cache_appends(tmp_path: Path):
    log = Logger(tmp_path)
    long_a = "abcdef" * 10
    long_b = "1234567890" * 6
    log.log_cache(0, "MISS", long_a)
    log.log_cache(1, "HIT", long_b)
    f = tmp_path / "logs" / "cache.log"
    assert f.read_text().splitlines() == [
        f"MISS shot_0 key={long_a[:12]}",
        f"HIT shot_1 key={long_b[:12]}",
    ]


def test_disabled_logger_is_noop(tmp_path: Path):
    log = Logger(tmp_path, enabled=False)
    log.log_call(
        label="x", instructions="i", prompt="p", output={}, output_type="X",
        latency_ms=1, model="m",
    )
    log.log_decision(0, {})
    log.log_cache(0, "HIT", "k")
    assert not (tmp_path / "logs").exists()


def test_stringify_prompt_truncates_binary_content():
    class _FakeBinary:
        def __init__(self, path: str) -> None:
            self.url = path

    out = _stringify_prompt(["hello", _FakeBinary("/x/y.png"), "world"])
    assert out == ["hello", "[_FakeBinary: /x/y.png]", "world"]
