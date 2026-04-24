"""Planner test using pydantic-ai's TestModel to avoid real network calls."""

from pathlib import Path

from opencanvas.agents import plan
from opencanvas.config import Settings
from opencanvas.schemas import Plan, Story


def test_planner_returns_plan_with_expected_shape(patched_llm, tmp_path: Path):
    story = Story(
        title="Test",
        shots=["Ada enters the warehouse.", "Ada finds the journal."],
    )
    result = plan(story, Settings(out_dir=tmp_path / "out"))

    assert isinstance(result, Plan)
    # Planner fallback guarantees shot count matches input
    assert len(result.shots) == 2
