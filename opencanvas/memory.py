from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

_SUBDIRS = ("characters", "characters_canonical", "locations", "props", "frames")
_SEP = "::"


def _dump_pairs(d: dict[tuple[str, str], Path], root: Path) -> dict[str, str]:
    return {f"{a}{_SEP}{b}": str(p.relative_to(root)) for (a, b), p in d.items()}


def _parse_pairs(d: dict, root: Path) -> dict[tuple[str, str], Path]:
    return {tuple(k.split(_SEP, 1)): root / v for k, v in d.items()}


def _parse_str_keys(d: dict, root: Path) -> dict[str, Path]:
    return {k: root / v for k, v in d.items()}


def _dump_str_keys(d: dict[object, Path], root: Path) -> dict[str, str]:
    return {str(k): str(v.relative_to(root)) for k, v in d.items()}


@dataclass
class Memory:
    """Visual state memory ℳ = (ℳ_c, ℳ_l, ℳ_o, ℳ_f) plus canonical anchors."""

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
        for sub in _SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)
        raw = json.loads(manifest.read_text())
        return cls(
            root=root,
            characters=_parse_pairs(raw.get("characters", {}), root),
            characters_canonical=_parse_pairs(raw.get("characters_canonical", {}), root),
            locations=_parse_str_keys(raw.get("locations", {}), root),
            props=_parse_pairs(raw.get("props", {}), root),
            frames={int(k): root / v for k, v in raw.get("frames", {}).items()},
        )

    def save(self) -> None:
        manifest = {
            "characters": _dump_pairs(self.characters, self.root),
            "characters_canonical": _dump_pairs(self.characters_canonical, self.root),
            "locations": _dump_str_keys(self.locations, self.root),
            "props": _dump_pairs(self.props, self.root),
            "frames": _dump_str_keys(self.frames, self.root),
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def _store(self, subdir: str, name: str, src: Path) -> Path:
        dest = self.root / subdir / f"{name}.png"
        if Path(src).resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        return dest

    def set_character(self, cid: str, state: str, src: Path) -> Path:
        self.characters[(cid, state)] = self._store("characters", f"{cid}__{state}", src)
        return self.characters[(cid, state)]

    def set_character_canonical(self, cid: str, state: str, src: Path) -> Path:
        self.characters_canonical[(cid, state)] = self._store(
            "characters_canonical", f"{cid}__{state}", src
        )
        return self.characters_canonical[(cid, state)]

    def set_location(self, lid: str, src: Path) -> Path:
        self.locations[lid] = self._store("locations", lid, src)
        return self.locations[lid]

    def set_prop(self, pid: str, state: str, src: Path) -> Path:
        self.props[(pid, state)] = self._store("props", f"{pid}__{state}", src)
        return self.props[(pid, state)]

    def set_frame(self, idx: int, src: Path) -> Path:
        self.frames[idx] = self._store("frames", f"shot_{idx:04d}", src)
        return self.frames[idx]

    def get_character(self, cid: str, state: str) -> Path | None:
        return self.characters.get((cid, state)) or self.characters_canonical.get((cid, state))

    def get_character_canonical(self, cid: str, state: str) -> Path | None:
        return self.characters_canonical.get((cid, state))

    def get_location(self, lid: str) -> Path | None:
        return self.locations.get(lid)

    def get_prop(self, pid: str, state: str) -> Path | None:
        return self.props.get((pid, state))

    def get_prop_any_state(self, pid: str) -> Path | None:
        for (p, _), path in reversed(self.props.items()):
            if p == pid:
                return path
        return None

    def get_frame(self, idx: int) -> Path | None:
        return self.frames.get(idx)
