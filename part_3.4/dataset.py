"""
dataset.py — StructuredDataset for cycle time prediction.

One row per case in the CSV/parquet. Pre-computes all padded tensors at load
time so __getitem__ is a single tensor index. Call prefetch_to_device(device)
to move everything to the GPU — then num_workers=0 and zero CPU-GPU transfer
per batch, eliminating the DataLoader bottleneck for small datasets.

Target: log1p(total_duration_sec) — scalar float32 per sample.
All prefixes of a case share the same target (case total duration is fixed).

Numeric slots (Dur, Amt, FW, NT, MC, CS, OA) are stored as log1p-transformed
floats in a separate numeric_values tensor rather than as vocabulary indices.
"""
from __future__ import annotations
import math, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from schema import EVENT_SLOTS
from vocab import Vocab, UNK_IDX

SLOT_NAMES         = [s[0] for s in EVENT_SLOTS]
NUMERIC_SLOT_NAMES = ["Dur", "Amt", "FW", "NT", "MC", "CS", "OA"]
NUMERIC_SLOTS      = frozenset(NUMERIC_SLOT_NAMES)
_N_NUMERIC         = len(NUMERIC_SLOT_NAMES)
_NUMERIC_IDX       = {slot: i for i, slot in enumerate(NUMERIC_SLOT_NAMES)}


class StructuredDataset(Dataset):
    """
    Parameters
    ----------
    data_path  : path to remaining_time CSV/parquet (one row per case)
    vocab      : fitted Vocab (fit on training split only, numeric slots excluded)
    split      : "train" | "val" | "test"
    max_events : maximum prefix length to consider
    min_prefix : skip prefixes shorter than this
    """

    def __init__(
        self,
        data_path:  str,
        vocab:      Vocab,
        split:      str   = "train",
        train_frac: float = 0.70,
        val_frac:   float = 0.15,
        max_events: int   = 50,
        min_prefix: int   = 1,
    ):
        self.vocab      = vocab
        self.max_events = max_events
        self.slot_names = SLOT_NAMES
        self.n_slots    = len(SLOT_NAMES)

        df = pd.read_parquet(data_path) if str(data_path).endswith(".parquet") else pd.read_csv(data_path)

        # Case-ordered train/val/test split
        case_ids = list(dict.fromkeys(df["case_id"].astype(str).tolist()))
        n        = len(case_ids)
        n_train  = int(n * train_frac)
        n_val    = int(n * val_frac)
        split_cases = {
            "train": set(case_ids[:n_train]),
            "val":   set(case_ids[n_train : n_train + n_val]),
            "test":  set(case_ids[n_train + n_val :]),
        }[split]
        df = df[df["case_id"].astype(str).isin(split_cases)].reset_index(drop=True)

        tok2idx = {slot: vocab.slots[slot]._tok2idx for slot in SLOT_NAMES if slot not in NUMERIC_SLOTS}

        # Encode case sequences
        sequences:  list[np.ndarray] = []
        num_arrays: list[np.ndarray] = []
        samples:    list[tuple[int, int, float]] = []

        for case_idx, (_, row) in enumerate(df.iterrows()):
            total_dur_sec = float(row["total_duration_sec"])
            n_events      = min(int(row["n_events"]), max_events)

            seq     = np.zeros((n_events, self.n_slots), dtype=np.int16)
            num_arr = np.zeros((n_events, _N_NUMERIC),   dtype=np.float32)

            for s, slot in enumerate(SLOT_NAMES):
                vals = str(row[slot]).split("|")[:n_events]
                if slot in NUMERIC_SLOTS:
                    ni = _NUMERIC_IDX[slot]
                    for e, val in enumerate(vals):
                        try:
                            num_arr[e, ni] = float(val)
                        except (ValueError, TypeError):
                            num_arr[e, ni] = 0.0
                else:
                    t2i = tok2idx[slot]
                    for e, val in enumerate(vals):
                        seq[e, s] = t2i.get(val, UNK_IDX)

            sequences.append(seq)
            num_arrays.append(num_arr)

            for prefix_len in range(min_prefix, n_events + 1):
                samples.append((case_idx, prefix_len, total_dur_sec))

        # Pre-compute all padded tensors — __getitem__ becomes a single index
        n_samples      = len(samples)
        slot_indices   = torch.zeros(n_samples, max_events, self.n_slots, dtype=torch.long)
        numeric_values = torch.zeros(n_samples, max_events, _N_NUMERIC,   dtype=torch.float32)
        padding_mask   = torch.ones( n_samples, max_events,               dtype=torch.bool)
        targets        = torch.zeros(n_samples,                           dtype=torch.float32)
        prefix_lens    = torch.zeros(n_samples,                           dtype=torch.long)

        for i, (case_idx, prefix_len, dur_sec) in enumerate(samples):
            seq     = sequences[case_idx][:prefix_len]
            num_arr = num_arrays[case_idx][:prefix_len]
            slot_indices[i,   :prefix_len] = torch.from_numpy(seq.astype(np.int64))
            numeric_values[i, :prefix_len] = torch.from_numpy(num_arr)
            padding_mask[i,   :prefix_len] = False
            targets[i]     = math.log1p(max(0.0, dur_sec))
            prefix_lens[i] = prefix_len

        self._slot_indices   = slot_indices
        self._numeric_values = numeric_values
        self._padding_mask   = padding_mask
        self._targets        = targets
        self._prefix_lens    = prefix_lens
        self._samples        = samples   # kept for train.py logging

    def prefetch_to_device(self, device: torch.device):
        """Move all tensors to device — eliminates CPU-GPU transfer per batch."""
        self._slot_indices   = self._slot_indices.to(device)
        self._numeric_values = self._numeric_values.to(device)
        self._padding_mask   = self._padding_mask.to(device)
        self._targets        = self._targets.to(device)
        self._prefix_lens    = self._prefix_lens.to(device)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "slot_indices":   self._slot_indices[idx],
            "numeric_values": self._numeric_values[idx],
            "padding_mask":   self._padding_mask[idx],
            "target":         self._targets[idx],
            "prefix_len":     self._prefix_lens[idx],
        }


def make_loader(
    dataset:     StructuredDataset,
    batch_size:  int,
    shuffle:     bool,
    num_workers: int = 4,
) -> DataLoader:
    on_gpu = dataset._slot_indices.is_cuda
    return DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = shuffle,
        num_workers        = 0 if on_gpu else num_workers,
        pin_memory         = not on_gpu,
        persistent_workers = False if on_gpu else (num_workers > 0),
        prefetch_factor    = None if on_gpu else (4 if num_workers > 0 else None),
    )
