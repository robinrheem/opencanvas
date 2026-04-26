from __future__ import annotations

from pathlib import Path

import typer

from .agents import plan_sync
from .config import Settings
from .pipeline import run_sync
from .schemas import Story

app = typer.Typer(help="OpenCanvas — CANVAS reproduction.")


def _load_story(path: Path) -> Story:
    return Story.model_validate_json(path.read_bytes())


def _overridden(**overrides) -> Settings:
    return Settings().model_copy(
        update={k: v for k, v in overrides.items() if v is not None}
    )


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
    settings = _overridden(
        out_dir=out_dir, k_candidates=k_candidates, model=model, seed=seed
    )
    plan, results = run_sync(story, settings)
    typer.echo(f"Plan: {len(plan.characters)} characters, {len(plan.locations)} locations")
    typer.echo(f"Generated {len(results)} shots → {settings.out_dir}")


@app.command(name="plan")
def plan_command(story_path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Run only the Global Planner."""
    typer.echo(plan_sync(_load_story(story_path), Settings()).model_dump_json(indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
