"""Regression tests for entity discovery + two-sided continuation validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

import opencanvas.agents as agents_mod
from opencanvas.agents import plan_sync
from opencanvas.config import Settings
from opencanvas.schemas import (
    Character,
    ContinuationMode,
    DiscoveredEntities,
    Story,
)


@pytest.fixture
def patched_llm_with_discovery(monkeypatch):
    """TestModel returns minimal entity discovery; everything else gets defaults."""
    def _factory(output_type, instructions, settings):
        if output_type is DiscoveredEntities:
            return Agent(
                TestModel(custom_output_args={
                    "characters": [{"id": "char-x", "name": "X", "description": "discovered"}],
                    "props": [],
                }),
                output_type=output_type,
                instructions=instructions,
            )
        return Agent(TestModel(), output_type=output_type, instructions=instructions)

    monkeypatch.setattr(agents_mod, "_agent_for", _factory)


def test_discovery_runs_when_entities_undeclared(patched_llm_with_discovery, tmp_path: Path):
    """Empty input story → planner runs entity discovery → Plan.characters non-empty."""
    story = Story(title="Smoke", shots=["Ada walks in.", "Ada finds a key."])
    plan = plan_sync(story, Settings(out_dir=tmp_path / "out"))

    # Discovery populated characters
    assert len(plan.characters) == 1
    assert plan.characters[0].id == "char-x"


def test_discovery_skipped_when_entities_declared(patched_llm_with_discovery, tmp_path: Path):
    """Author-declared entities → discovery is bypassed; declared list is preserved."""
    story = Story(
        title="Authored",
        shots=["Bob enters.", "Bob exits."],
        characters=[Character(id="char-bob", name="Bob", description="man in coat")],
    )
    plan = plan_sync(story, Settings(out_dir=tmp_path / "out"))

    # No discovery; declared character preserved
    assert len(plan.characters) == 1
    assert plan.characters[0].id == "char-bob"
