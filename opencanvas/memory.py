from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

_SUBDIRS = ("characters", "characters_canonical", "locations", "props", "frames")
_TUPLE_SEP = "::"


def _dump_tuple_dict(d: dict[tuple[str, str], Path], root: Path) -> dict[str, str]:
    return {f"{a}{_TUPLE_SEP}{b}": str(p.relative_to(root)) for (a, b), p in d.items()}


def _parse_tuple_dict(d: dict, root: Path) -> dict[tuple[str, str], Path]:
    return {tuple(k.split(_TUPLE_SEP, 1)): root / v for k, v in d.items()}


@dataclass
class Memory:
    """Paper's visual state memory ℳ = (ℳ_c, ℳ_l, ℳ_o, ℳ_f).

    Canonical character anchors live separately and are never overwritten by
    `set_character`.
    """

    root: Path
    characters: dict[tuple[str, str], Path] = field(default_factory=dict)
    characters_canonical: dict[tuple[str, str], Path] = field(default_factory=dict)
    locations: dict[str, Path] = field(default_factory=dict)
    props: dict[tuple[str, str], Path] = field(default_factory=dict)
    frames: dict[int, Path] = field(default_factory=dict)

    @classmethod
    def empty(cls, root: Path) -> "Memory":
        root = Path(root)
        for sub in _SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    @classmethod
    def load_or_empty(cls, root: Path) -> "Memory":
        root = Path(root)
        manifest = root / "manifest.json"
        if not manifest.exists():
            return cls.empty(root)
        raw = json.loads(manifest.read_text())
        for sub in _SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            characters=_parse_tuple_dict(raw.get("characters", {}), root),
            characters_canonical=_parse_tuple_dict(raw.get("characters_canonical", {}), root),
            locations={k: root / v for k, v in raw.get("locations", {}).items()},
            props=_parse_tuple_dict(raw.get("props", {}), root),
            frames={int(k): root / v for k, v in raw.get("frames", {}).items()},
        )

    def save(self) -> None:
        manifest = {
            "characters": _dump_tuple_dict(self.characters, self.root),
            "characters_canonical": _dump_tuple_dict(self.characters_canonical, self.root),
            "locations": {k: str(v.relative_to(self.root)) for k, v in self.locations.items()},
            "props": _dump_tuple_dict(self.props, self.root),
            "frames": {str(k): str(v.relative_to(self.root)) for k, v in self.frames.items()},
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def _store(self, subdir: str, name: str, src: Path) -> Path:
        dest = self.root / subdir / f"{name}.png"
        if Path(src).resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        return dest

    def set_character(self, character_id: str, state: str, path: Path) -> Path:
        self.characters[(character_id, state)] = self._store(
            "characters", f"{character_id}__{state}", path
        )
        return self.characters[(character_id, state)]

    def set_character_canonical(self, character_id: str, state: str, path: Path) -> Path:
        self.characters_canonical[(character_id, state)] = self._store(
            "characters_canonical", f"{character_id}__{state}", path
        )
        return self.characters_canonical[(character_id, state)]

    def set_location(self, location_id: str, path: Path) -> Path:
        self.locations[location_id] = self._store("locations", location_id, path)
        return self.locations[location_id]

    def set_prop(self, prop_id: str, state: str, path: Path) -> Path:
        self.props[(prop_id, state)] = self._store("props", f"{prop_id}__{state}", path)
        return self.props[(prop_id, state)]

    def set_frame(self, shot_index: int, path: Path) -> Path:
        self.frames[shot_index] = self._store("frames", f"shot_{shot_index:04d}", path)
        return self.frames[shot_index]

    def get_character(self, character_id: str, state: str) -> Path | None:
        return self.characters.get((character_id, state)) or self.characters_canonical.get(
            (character_id, state)
        )

    def get_character_canonical(self, character_id: str, state: str) -> Path | None:
        return self.characters_canonical.get((character_id, state))

    def get_location(self, location_id: str) -> Path | None:
        return self.locations.get(location_id)

    def get_prop(self, prop_id: str, state: str) -> Path | None:
        return self.props.get((prop_id, state))

    def get_frame(self, shot_index: int) -> Path | None:
        return self.frames.get(shot_index)
