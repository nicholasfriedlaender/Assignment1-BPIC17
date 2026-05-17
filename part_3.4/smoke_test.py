"""
smoke_test.py — Fast end-to-end check of the remaining_time pipeline.

Samples 200 cases from the real logs_filtered.csv and runs the full pipeline:
preprocess → vocab → dataset → model → train step.

Usage:
    python smoke_test.py
"""
import sys, tempfile, shutil
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))   # ensure remaining_time/ modules take priority

REAL_LOG = Path(__file__).parent / "logs_filtered.csv"
N_CASES  = 200   # sample size — fast but realistic


def sample_real_data(n_cases: int) -> pd.DataFrame:
    df       = pd.read_csv(REAL_LOG)
    case_col = "case:concept:name"
    cases    = df[case_col].drop_duplicates().sample(n=n_cases, random_state=42)
    return df[df[case_col].isin(cases)].reset_index(drop=True)


def run():
    tmp = Path(tempfile.mkdtemp())
    try:
        print("── Smoke test: remaining_time pipeline ─────────────────────────────")
        print(f"   Using real data: {REAL_LOG.name}  (sample: {N_CASES} cases)")

        # ── 1. Preprocess ──────────────────────────────────────────────────────
        print("\n[1/5] preprocess …", end=" ", flush=True)
        raw_csv = tmp / "sample.csv"
        sample_real_data(N_CASES).to_csv(raw_csv, index=False)

        from preprocess import preprocess
        csv_path = tmp / "remaining_time.csv"
        df = preprocess(raw_csv, csv_path, max_events=10)
        assert len(df) > 0, "preprocess produced no rows"
        assert "A" in df.columns and "total_duration_sec" in df.columns, \
            f"missing columns — got {list(df.columns)}"
        assert df["total_duration_sec"].ge(0).all(), "negative durations found"
        # Dur column should now contain floats, not bin strings
        sample_dur = df["Dur"].iloc[0].split("|")[0]
        try:
            float(sample_dur)
        except ValueError:
            raise AssertionError(f"Dur column should contain floats, got: {sample_dur!r}")
        print(f"OK  ({len(df)} cases, Dur[0]={sample_dur})")

        # ── 2. Vocabulary ──────────────────────────────────────────────────────
        print("[2/5] vocab fit/save/load …", end=" ", flush=True)
        from vocab import Vocab
        from dataset import NUMERIC_SLOTS, NUMERIC_SLOT_NAMES, SLOT_NAMES
        slot_names = SLOT_NAMES
        vocab      = Vocab.fit(df, slot_names, numeric_slots=NUMERIC_SLOTS)
        vocab_path = tmp / "vocab.json"
        vocab.save(vocab_path)
        vocab2 = Vocab.load(vocab_path)
        assert vocab.vocab_sizes() == vocab2.vocab_sizes(), "vocab round-trip mismatch"
        assert "Dur" not in vocab.vocab_sizes(), "Dur should not have a vocab entry (it's numeric)"
        assert "Amt" not in vocab.vocab_sizes(), "Amt should not have a vocab entry (it's numeric)"
        print(f"OK  ({len(vocab.slot_names)} categorical slots, sizes: {vocab.vocab_sizes()})")

        # ── 3. Dataset ─────────────────────────────────────────────────────────
        print("[3/5] dataset …", end=" ", flush=True)
        from dataset import StructuredDataset, make_loader
        ds = StructuredDataset(
            data_path  = str(csv_path),
            vocab      = vocab,
            split      = "train",
            train_frac = 0.70,
            val_frac   = 0.15,
            max_events = 10,
        )
        assert len(ds) > 0, "dataset is empty"
        sample = ds[0]
        assert sample["slot_indices"].shape == (10, len(slot_names)), \
            f"unexpected shape {sample['slot_indices'].shape}"
        assert sample["padding_mask"].dtype == torch.bool
        assert "target" in sample, "missing target key"
        assert sample["target"].dtype == torch.float32, f"target dtype {sample['target'].dtype}"
        assert sample["target"].item() >= 0.0, "log1p target must be non-negative"
        assert "numeric_values" in sample, "missing numeric_values key"
        assert sample["numeric_values"].shape == (10, len(NUMERIC_SLOT_NAMES)), \
            f"unexpected numeric_values shape {sample['numeric_values'].shape}"
        assert sample["numeric_values"].dtype == torch.float32

        loader = make_loader(ds, batch_size=8, shuffle=True, num_workers=0)
        batch  = next(iter(loader))
        assert batch["slot_indices"].shape[0] <= 8
        assert "target" in batch
        assert batch["target"].shape == (batch["slot_indices"].shape[0],), \
            f"target shape mismatch: {batch['target'].shape}"
        assert "numeric_values" in batch
        assert batch["numeric_values"].shape == (batch["slot_indices"].shape[0], 10, len(NUMERIC_SLOT_NAMES))
        print(f"OK  ({len(ds)} samples, slot_indices={tuple(batch['slot_indices'].shape)}, "
              f"numeric_values={tuple(batch['numeric_values'].shape)})")

        # ── 4. Model forward + ablation ────────────────────────────────────────
        print("[4/5] model forward + ablation …", end=" ", flush=True)
        from model import StructuredModel, RemainingTimeLoss
        model = StructuredModel(
            vocab_sizes        = vocab.vocab_sizes(),
            slot_names         = slot_names,
            numeric_slot_names = NUMERIC_SLOT_NAMES,
            d_model            = 64,
            n_heads            = 4,
            n_layers           = 2,
            ffn_dim            = 128,
            max_events         = 10,
            n_outcomes         = 1,
        )
        assert len(model.numeric_projs) == len(NUMERIC_SLOT_NAMES), \
            f"expected {len(NUMERIC_SLOT_NAMES)} numeric projections"
        assert "Dur" not in model.embeddings, "Dur should use numeric_projs, not embeddings"

        preds = model(batch["slot_indices"], batch["padding_mask"],
                      numeric_values=batch["numeric_values"])
        assert preds.shape == (batch["slot_indices"].shape[0],), \
            f"expected scalar output per sample, got {preds.shape}"

        # Ablation: zeroing a slot must change the output
        preds_ablated = model(batch["slot_indices"], batch["padding_mask"],
                              numeric_values=batch["numeric_values"],
                              ablate_slots=frozenset(["A"]))
        assert not torch.allclose(preds, preds_ablated), \
            "ablation had no effect — slot A may be all-zero already"

        criterion = RemainingTimeLoss()
        loss, metrics = criterion(preds, batch["target"])
        assert 0.0 <= loss.item() < 1000.0, f"loss out of range: {loss.item()}"
        assert "mae_hrs" in metrics
        assert metrics["mae_sec"] >= 0.0
        print(f"OK  (loss={loss.item():.4f}  mae_hrs={metrics['mae_hrs']:.2f})")

        # ── 5. Training step ───────────────────────────────────────────────────
        print("[5/5] training step (forward + backward + optimiser) …", end=" ", flush=True)
        from torch.optim import AdamW
        opt = AdamW(model.parameters(), lr=1e-3)
        model.train()
        opt.zero_grad()
        preds2 = model(batch["slot_indices"], batch["padding_mask"],
                       numeric_values=batch["numeric_values"])
        loss2, _ = criterion(preds2, batch["target"])
        loss2.backward()
        opt.step()
        # Weights should have changed
        with torch.no_grad():
            preds3 = model(batch["slot_indices"], batch["padding_mask"],
                           numeric_values=batch["numeric_values"])
        assert not torch.allclose(preds2.detach(), preds3), \
            "weights did not update after optimiser step"
        print("OK")

        print("\n✓ All checks passed.\n")
        print("── Next steps ───────────────────────────────────────────────────────")
        print("  # Step 1 — preprocess raw log (once)")
        print("  python preprocess.py ../logs_filtered.csv data/remaining_time.parquet")
        print()
        print("  # Step 2 — train")
        print("  python train.py --data data/remaining_time.parquet")
        print()
        print("  # Step 3 — evaluate + slot importance")
        print("  python evaluate.py --data data/remaining_time.parquet")
        print()

    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    run()
