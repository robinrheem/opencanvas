"""Per-run observability: JSONL of every LLM/VLM call, per-shot decision
summary, cache hit/miss audit. Lives under settings.out_dir/logs/.
"""

from __future__ import annotations

import contextvars
import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Active Logger for the current pipeline run. agents._llm reads this so we
# don't have to thread `logger` through every planner sub-call.
_current: contextvars.ContextVar["Logger | None"] = contextvars.ContextVar(
    "_current_logger", default=None,
)


def set_active(logger: "Logger | None") -> None:
    _current.set(logger)


def active() -> "Logger | None":
    return _current.get()


class Logger:
    """Append-only sink. No-op when `enabled=False` (tests / benchmarks)."""

    def __init__(self, out_dir: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.root = Path(out_dir) / "logs"
        if enabled:
            (self.root / "llm").mkdir(parents=True, exist_ok=True)
            (self.root / "decisions").mkdir(parents=True, exist_ok=True)

    def log_call(
        self, label: str, instructions: str, prompt: Any, output: Any,
        output_type: str, latency_ms: int, model: str,
    ) -> None:
        if not self.enabled:
            return
        line = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "model": model,
            "instructions": instructions,
            "user": _stringify_prompt(prompt),
            "output_type": output_type,
            "output": _to_jsonable(output),
            "latency_ms": latency_ms,
        }
        path = self.root / "llm" / f"{label}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def log_decision(self, shot_index: int, payload: dict) -> None:
        if not self.enabled:
            return
        path = self.root / "decisions" / f"shot_{shot_index:04d}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    def log_cache(self, shot_index: int, status: str, key: str) -> None:
        if not self.enabled:
            return
        path = self.root / "cache.log"
        with path.open("a") as f:
            f.write(f"{status} shot_{shot_index} key={key[:12]}\n")


def _stringify_prompt(prompt: Any) -> Any:
    """Replace pydantic_ai BinaryContent objects with their underlying file
    path so logs stay human-readable (no megabyte base64 payloads)."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return [_stringify_prompt(p) for p in prompt]
    cls = type(prompt).__name__
    for attr in ("url", "path", "filename", "data_url"):
        v = getattr(prompt, attr, None)
        if v:
            return f"[{cls}: {v}]"
    return f"[{cls}]"


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj
