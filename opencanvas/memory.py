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


@dataclass
class Memory:
    """Visual state memory ℳ = (ℳ_c, ℳ_l, ℳ_o, ℳ_f) plus canonical anchors.

    Direct dict access for simple lookups: `memory.locations.get(lid)`,
    `memory.frames.get(idx)`, etc. Use `memory.character(...)` /
    `memory.prop(...)` for smart fallback chains.
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
        for sub in _SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)
        raw = json.loads(manifest.read_text())
        return cls(
            root=root,
            characters=_parse_pairs(raw.get("characters", {}), root),
            characters_canonical=_parse_pairs(raw.get("characters_canonical", {}), root),
            locations={k: root / v for k, v in raw.get("locations", {}).items()},
            props=_parse_pairs(raw.get("props", {}), root),
            frames={int(k): root / v for k, v in raw.get("frames", {}).items()},
        )

    def save(self) -> None:
        manifest = {
            "characters": _dump_pairs(self.characters, self.root),
            "characters_canonical": _dump_pairs(self.characters_canonical, self.root),
            "locations": {k: str(v.relative_to(self.root)) for k, v in self.locations.items()},
            "props": _dump_pairs(self.props, self.root),
            "frames": {str(k): str(v.relative_to(self.root)) for k, v in self.frames.items()},
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def _copy_in(self, subdir: str, name: str, src: Path) -> Path:
        dest = self.root / subdir / f"{name}.png"
        if Path(src).resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        return dest

    # --- Mutators (idempotent: copy file into memory dir + record path) ---

    def add_character(self, cid: str, state: str, src: Path) -> Path:
        path = self._copy_in("characters", f"{cid}__{state}", src)
        self.characters[(cid, state)] = path
        return path

    def add_canonical(self, cid: str, state: str, src: Path) -> Path:
        path = self._copy_in("characters_canonical", f"{cid}__{state}", src)
        self.characters_canonical[(cid, state)] = path
        return path

    def add_location(self, lid: str, src: Path) -> Path:
        path = self._copy_in("locations", lid, src)
        self.locations[lid] = path
        return path

    def add_prop(self, pid: str, state: str, src: Path) -> Path:
        path = self._copy_in("props", f"{pid}__{state}", src)
        self.props[(pid, state)] = path
        return path

    def add_frame(self, idx: int, src: Path) -> Path:
        path = self._copy_in("frames", f"shot_{idx:04d}", src)
        self.frames[idx] = path
        return path

    # --- Smart lookups (logic beyond plain dict.get) ---

    def character(self, cid: str, state: str) -> Path | None:
        """Recent state anchor; canonical fallback. (Algorithm 2.)"""
        return self.characters.get((cid, state)) or self.characters_canonical.get((cid, state))

    def prop(self, pid: str, state: str) -> Path | None:
        """State match → 'default' state → any prior state of this prop."""
        return (
            self.props.get((pid, state))
            or self.props.get((pid, "default"))
            or self.prop_any_state(pid)
        )

    def prop_any_state(self, pid: str) -> Path | None:
        for (p, _), path in reversed(self.props.items()):
            if p == pid:
                return path
        return None
