"""Action space — soul.md §2.3."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class ActionKind(Enum):
    PROPOSE = "propose"
    SELECT_DATA = "select_data"
    TRAIN = "train"
    CURATE = "curate"
    INVESTIGATE = "investigate"
    REFRAIN = "refrain"


@dataclass
class Action:
    kind: ActionKind
    payload: Dict[str, Any] = field(default_factory=dict)
