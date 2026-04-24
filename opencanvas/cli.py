from __future__ import annotations

from pathlib import Path

import typer

from .config import Settings
from .pipeline import run as run_pipeline
from .schemas import Story

app = typer.Typer(help="OpenCanvas — CANVAS reproduction.")


def _load_story(path: Path) -> Story:
    return Story.model_validate_json(path.read_bytes())


def _settings_with_overrides(**overrides) -> Settings:
    base = Settings()
    return base.model_copy(update={k: v for k, v in overrides.items() if v is not None})


@app.command()
def run(
    story_path: Path = typer.Argument(..., exists=True, readable=True),
    out_dir: Path | None = typer.Option(None, "--out", "-o", help="Output directory."),
    k_candidates: int | None = typer.Option(None, "--k", help="Candidates per shot."),
    model: str | None = typer.Option(None, "--model", help="Override LLM model."),
    seed: int | None = typer.Option(None, "--seed", help="Base seed."),
) -> None:
    """Generate a storyboard for STORY_PATH."""
    story = _load_story(story_path)
    settings = _settings_with_overrides(
        out_dir=out_dir, k_candidates=k_candidates, model=model, seed=seed
    )
    plan, results = run_pipeline(story, settings)
    typer.echo(f"Plan: {len(plan.characters)} characters, {len(plan.locations)} locations")
    typer.echo(f"Generated {len(results)} shots → {settings.out_dir}")


@app.command(name="plan")
def plan_only(story_path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Run only the Global Planner."""
    from .agents import plan as plan_agent

    typer.echo(plan_agent(_load_story(story_path), Settings()).model_dump_json(indent=2))


if __name__ == "__main__":
    app()
