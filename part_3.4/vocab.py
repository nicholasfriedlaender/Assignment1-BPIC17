"""
vocab.py — Per-slot string → integer vocabulary.

Special indices: PAD=0, UNK=1, known values start at 2.
"""
from __future__ import annotations
import json
from pathlib import Path

PAD_IDX = 0
UNK_IDX = 1


class SlotVocab:
    def __init__(self, values: list[str]):
        self._tok2idx: dict[str, int] = {v: i + 2 for i, v in enumerate(sorted(set(values)))}
        self._idx2tok: dict[int, str] = {i: v for v, i in self._tok2idx.items()}
        self.size = len(self._tok2idx) + 2  # +2 for PAD and UNK

    def encode(self, value: str) -> int:
        return self._tok2idx.get(str(value), UNK_IDX)

    def decode(self, idx: int) -> str:
        if idx == PAD_IDX: return "<PAD>"
        if idx == UNK_IDX: return "<UNK>"
        return self._idx2tok.get(idx, "<UNK>")

    def to_dict(self) -> dict:
        return {"tok2idx": self._tok2idx}

    @classmethod
    def from_dict(cls, d: dict) -> "SlotVocab":
        obj = cls.__new__(cls)
        obj._tok2idx = d["tok2idx"]
        obj._idx2tok = {i: v for v, i in obj._tok2idx.items()}
        obj.size = len(obj._tok2idx) + 2
        return obj


class Vocab:
    """Collection of per-slot vocabularies."""

    def __init__(self, slots: Dict[str, SlotVocab]):
        self.slots = slots

    @classmethod
    def fit(cls, df, slot_names: list[str], numeric_slots: frozenset = frozenset()) -> "Vocab":
        """Fit from training-split dataframe (pipe-separated string columns)."""
        slot_vocabs = {}
        for slot in slot_names:
            if slot in numeric_slots or slot not in df.columns:
                continue
            all_vals = []
            for cell in df[slot].dropna().astype(str):
                all_vals.extend(v for v in cell.split("|") if v not in ("NaN", "nan", ""))
            slot_vocabs[slot] = SlotVocab(all_vals)
        return cls(slot_vocabs)

    def save(self, path: Path):
        Path(path).write_text(json.dumps(
            {slot: sv.to_dict() for slot, sv in self.slots.items()}, indent=2
        ))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        data = json.loads(Path(path).read_text())
        return cls({slot: SlotVocab.from_dict(d) for slot, d in data.items()})

    @property
    def slot_names(self) -> list[str]:
        return list(self.slots.keys())

    def vocab_sizes(self) -> dict[str, int]:
        return {s: sv.size for s, sv in self.slots.items()}
