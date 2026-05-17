"""
evaluate.py — Evaluation + slot importance for the cycle time model.

Outputs:
  1. Overall MAE (sec/hrs), RMSE, MAPE
  2. Slot ablation importance  (ablated MAE - baseline MAE per slot)
  3. MAE by prefix length

Usage:
    python evaluate.py \
        --data data/remaining_time.parquet \
        --checkpoint checkpoints_remaining_time/best_model.pt
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vocab import Vocab
from dataset import StructuredDataset, make_loader
from model import StructuredModel



DEFAULT_BUCKETS: list[tuple[str, range]] = [
    ("early", range(1,  6)),   # prefix lengths 1–5
    ("mid",   range(6,  16)),  # prefix lengths 6–15
    ("late",  range(16, 51)),  # prefix lengths 16–50
]


@torch.no_grad()
def eval_pass(
    model:         StructuredModel,
    loader:        DataLoader,
    device:        torch.device,
    use_bf16:      bool,
    ablate_slots:  frozenset[str] = frozenset(),
    prefix_filter: frozenset[int] | None = None,
) -> dict:
    model.eval()
    preds_sec   = []
    targets_sec = []
    by_prefix   = defaultdict(lambda: {"preds": [], "targets": []})

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float32,
                                  enabled=(use_bf16 and device.type == "cuda"))

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with amp_ctx:
            output = model(
                batch["slot_indices"],
                batch["padding_mask"],
                numeric_values=batch.get("numeric_values"),
                ablate_slots=ablate_slots,
            )

        pred_s   = torch.expm1(output.float().clamp(min=0)).cpu().tolist()
        target_s = torch.expm1(batch["target"].float().clamp(min=0)).cpu().tolist()
        pl_list  = batch["prefix_len"].cpu().tolist()

        for ps, ts, pl in zip(pred_s, target_s, pl_list):
            by_prefix[int(pl)]["preds"].append(ps)
            by_prefix[int(pl)]["targets"].append(ts)
            if prefix_filter is None or int(pl) in prefix_filter:
                preds_sec.append(ps)
                targets_sec.append(ts)

    n = max(len(preds_sec), 1)
    p_arr = np.array(preds_sec)
    t_arr = np.array(targets_sec)
    mae_sec  = float(np.mean(np.abs(p_arr - t_arr)))
    rmse_sec = float(np.sqrt(np.mean((p_arr - t_arr) ** 2)))
    mape_pct = float(np.mean(np.abs(p_arr - t_arr) / np.maximum(t_arr, 1.0) * 100))

    by_prefix_out = {}
    for pl, v in sorted(by_prefix.items()):
        p_pl    = np.array(v["preds"])
        t_pl    = np.array(v["targets"])
        errs_pl = np.abs(p_pl - t_pl)
        mae_pl  = float(np.mean(errs_pl))
        by_prefix_out[pl] = {
            "mae_sec":  round(mae_pl, 1),
            "mae_hrs":  round(mae_pl / 3600, 4),
            "rmse_sec": round(float(np.sqrt(np.mean((p_pl - t_pl) ** 2))), 1),
            "mape_pct": round(float(np.mean(errs_pl / np.maximum(t_pl, 1.0) * 100)), 2),
            "n":        len(p_pl),
        }

    return {
        "mae_sec":   round(mae_sec, 1),
        "mae_hrs":   round(mae_sec / 3600, 4),
        "rmse_sec":  round(rmse_sec, 1),
        "mape_pct":  round(mape_pct, 2),
        "n_samples": n,
        "by_prefix": by_prefix_out,
    }



def slot_importance(model, loader, device, use_bf16, slot_names, n_repeats: int = 5) -> dict[str, float]:
    baseline = eval_pass(model, loader, device, use_bf16)["mae_sec"]
    print(f"\n  Baseline MAE: {baseline:.0f} sec  ({baseline/3600:.2f} hrs)")
    importances = {}
    for slot in slot_names:
        # Average over multiple permutations to reduce variance
        ablated_mae = float(np.mean([
            eval_pass(model, loader, device, use_bf16, ablate_slots=frozenset([slot]))["mae_sec"]
            for _ in range(n_repeats)
        ]))
        importances[slot] = round(ablated_mae - baseline, 1)   # positive = slot is important
        increase = importances[slot]
        bar      = "█" * max(0, int(abs(increase) / max(baseline, 1) * 300))
        print(f"  {slot:<5} ablated MAE={ablated_mae:.0f}s  increase={increase:+.0f}s  {bar}")
    return importances


def slot_importance_by_prefix(
    model:      StructuredModel,
    loader:     DataLoader,
    device:     torch.device,
    use_bf16:   bool,
    slot_names: list[str],
    buckets:    list[tuple[str, range]] = DEFAULT_BUCKETS,
    n_repeats:  int = 5,
) -> dict[str, dict[str, float]]:
    """Feature importance per prefix-length bucket (MAE delta from permutation ablation)."""
    results: dict[str, dict[str, float]] = {}

    for label, bucket_range in buckets:
        pf = frozenset(bucket_range)
        baseline = eval_pass(model, loader, device, use_bf16, prefix_filter=pf)
        n_samples = baseline["n_samples"]
        if n_samples == 0:
            print(f"\n  [{label}] No samples — skipping")
            continue
        baseline_mae = baseline["mae_sec"]
        print(f"\n  [{label}] n={n_samples}  Baseline MAE: {baseline_mae:.0f}s ({baseline_mae/3600:.2f}hrs)")

        bucket_importances: dict[str, float] = {}
        for slot in slot_names:
            ablated_mae = float(np.mean([
                eval_pass(model, loader, device, use_bf16,
                          ablate_slots=frozenset([slot]), prefix_filter=pf)["mae_sec"]
                for _ in range(n_repeats)
            ]))
            importance = round(ablated_mae - baseline_mae, 1)
            bucket_importances[slot] = importance
            bar = "█" * max(0, int(abs(importance) / max(baseline_mae, 1) * 300))
            print(f"    {slot:<5} ablated={ablated_mae:.0f}s  delta={importance:+.0f}s  {bar}")

        results[label] = bucket_importances

    return results



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt     = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab    = Vocab.load(Path(args.checkpoint).parent / "vocab.json")
    use_bf16 = args.bf16 and device.type == "cuda"

    model = StructuredModel(
        vocab_sizes        = vocab.vocab_sizes(),
        slot_names         = ckpt["slot_names"],
        numeric_slot_names = ckpt.get("numeric_slot_names", []),
        d_model            = ckpt["d_model"],
        n_heads            = ckpt["n_heads"],
        n_layers           = ckpt["n_layers"],
        ffn_dim            = ckpt["ffn_dim"],
        dropout            = 0.0,
        max_events         = ckpt["max_events"],
        n_outcomes         = ckpt.get("n_outcomes", 1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}  val_mae={ckpt['val_mae_sec']:.0f}s  ({ckpt['val_mae_sec']/3600:.2f}hrs)")

    test_ds = StructuredDataset(
        data_path  = args.data,
        vocab      = vocab,
        split      = "test",
        max_events = ckpt["max_events"],
    )
    if device.type == "cuda":
        test_ds.prefetch_to_device(device)
    loader = make_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.workers)
    print(f"Test samples: {len(test_ds):,}")

    results = eval_pass(model, loader, device, use_bf16)
    print(f"\n── Overall ──────────────────────────────────────────────────────────")
    print(f"  MAE          : {results['mae_sec']:.0f} sec  ({results['mae_hrs']:.2f} hrs)")
    print(f"  RMSE         : {results['rmse_sec']:.0f} sec")
    print(f"  MAPE         : {results['mape_pct']:.1f}%")
    print(f"  Samples      : {results['n_samples']:,}")

    print(f"\n── MAE by prefix length ─────────────────────────────────────────────")
    print(f"  {'prefix':<8} {'n':<8} {'MAE (sec)':>10} {'MAE (hrs)':>10} {'RMSE':>10} {'MAPE%':>8}")
    print(f"  {'-'*58}")
    for pl, v in results["by_prefix"].items():
        print(f"  {pl:<8} {v['n']:<8} {v['mae_sec']:>10.0f} {v['mae_hrs']:>10.2f} "
              f"{v['rmse_sec']:>10.0f} {v['mape_pct']:>8.1f}")

    print(f"\n── Slot importance (MAE increase when slot permuted) ────────────────")
    importances = slot_importance(model, loader, device, use_bf16, ckpt["slot_names"])
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Ranking:")
    for rank, (slot, imp) in enumerate(ranked, 1):
        bar = "█" * max(0, int(imp / max(results["mae_sec"], 1) * 300))
        print(f"  {rank:2}. {slot:<5}  {imp:+.0f}s  {bar}")

    print(f"\n── Slot importance by prefix bucket ─────────────────────────────────")
    importance_by_bucket = slot_importance_by_prefix(
        model, loader, device, use_bf16, ckpt["slot_names"]
    )
    labels = [lab for lab, _ in DEFAULT_BUCKETS]
    print(f"\n  {'slot':<6}" + "".join(f"{lab:>12}" for lab in labels))
    print("  " + "-" * (6 + 12 * len(labels)))
    for slot in ckpt["slot_names"]:
        row = f"  {slot:<6}"
        for lab in labels:
            imp = importance_by_bucket.get(lab, {}).get(slot, 0.0)
            row += f"{imp:>+11.0f}s"
        print(row)

    out = Path(args.checkpoint).parent / "eval_results.json"
    results["slot_importance"] = importances
    results["slot_importance_by_bucket"] = importance_by_bucket
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",        required=True)
    p.add_argument("--checkpoint",  default="checkpoints_remaining_time/best_model.pt")
    p.add_argument("--batch_size",  type=int,  default=2048)
    p.add_argument("--workers",     type=int,  default=4)
    p.add_argument("--bf16",        action="store_true", default=True)
    p.add_argument("--no_bf16",     dest="bf16", action="store_false")
    main(p.parse_args())
