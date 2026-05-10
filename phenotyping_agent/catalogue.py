"""Phenotype catalogue — soul.md §7.

Append-only JSONL store. Each accepted specimen is one line. The catalogue
is the externalised body of the agent.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Specimen:
    arch_spec: Dict[str, Any]
    training: Dict[str, Any]
    capability: Dict[str, float]
    plausibility: Dict[str, float]
    topography: Dict[str, Any] = field(default_factory=dict)
    hierarchy: Dict[str, Any] = field(default_factory=dict)
    latent: Dict[str, Any] = field(default_factory=dict)
    neural_alignment: Dict[str, float] = field(default_factory=dict)
    free_energy: Optional[float] = None
    note: str = ""
    accepted_at: float = field(default_factory=time.time)


class Catalogue:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, specimen: Specimen) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(specimen), sort_keys=True) + "\n")

    def load(self) -> List[Specimen]:
        if not self.path.exists():
            return []
        out: List[Specimen] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(Specimen(**json.loads(line)))
        return out
