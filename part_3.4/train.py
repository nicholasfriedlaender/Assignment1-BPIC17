"""
train.py — Train the structured embedding model for cycle time prediction.

Optimised for RTX 4090:
  - BF16 mixed precision
  - torch.compile()
  - Fused AdamW
  - Large batch sizes (model is ~3M params vs ModernBERT's 150M)

Usage:
    # Step 1: preprocess (once)
    python preprocess.py ../BPIC2017.csv data/remaining_time.parquet

    # Step 2: train
    python train.py --data data/remaining_time.parquet
"""
from __future__ import annotations
import os, argparse, json, math, time, random
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from schema import EVENT_SLOTS
from vocab import Vocab
from dataset import StructuredDataset, make_loader, NUMERIC_SLOTS, NUMERIC_SLOT_NAMES
from model import StructuredModel, RemainingTimeLoss


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimiser, scheduler, device, training, use_bf16, grad_clip, desc=""):
    model.train(training)
    totals, n = {}, 0

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float32,
                                  enabled=(use_bf16 and device.type == "cuda"))
    no_grad = torch.no_grad() if not training else torch.enable_grad()

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in pbar:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with no_grad, amp_ctx:
            preds = model(
                slot_indices   = batch["slot_indices"],
                padding_mask   = batch["padding_mask"],
                numeric_values = batch.get("numeric_values"),
            )
            loss, metrics = criterion(preds, batch["target"])

        if training:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            scheduler.step()

        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
        pbar.set_postfix(loss=f"{totals['loss']/n:.4f}", mae_hrs=f"{totals['mae_hrs']/n:.2f}", refresh=False)

    return {k: v / max(n, 1) for k, v in totals.items()}


def main(args: argparse.Namespace):
    set_seed(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = args.bf16 and device.type == "cuda"
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"BF16: {'ON' if use_bf16 else 'OFF'}")

    # ── Fit vocabulary from training split ────────────────────────────────────
    slot_names = [s[0] for s in EVENT_SLOTS]
    print("\nFitting vocabulary on training split …")
    df_raw    = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
    case_ids  = list(dict.fromkeys(df_raw["case_id"].astype(str).tolist()))
    n_train   = int(len(case_ids) * 0.70)
    train_df  = df_raw[df_raw["case_id"].astype(str).isin(set(case_ids[:n_train]))]
    vocab = Vocab.fit(train_df, slot_names, numeric_slots=NUMERIC_SLOTS)
    vocab.save(out_dir / "vocab.json")
    print(f"Vocabulary saved → {out_dir / 'vocab.json'}")
    for slot, size in vocab.vocab_sizes().items():
        print(f"  {slot:<4} {size} values")

    print("\nBuilding datasets …")
    common   = dict(data_path=args.data, vocab=vocab, max_events=args.max_events, min_prefix=args.min_prefix)
    train_ds = StructuredDataset(**common, split="train")
    val_ds   = StructuredDataset(**common, split="val")
    test_ds  = StructuredDataset(**common, split="test")

    print(f"  train: {len(train_ds):,} samples  ({len(set(s[0] for s in train_ds._samples)):,} cases)")
    print(f"  val:   {len(val_ds):,} samples  ({len(set(s[0] for s in val_ds._samples)):,} cases)")
    print(f"  test:  {len(test_ds):,} samples  ({len(set(s[0] for s in test_ds._samples)):,} cases)")

    # Prefix length distribution (train)
    prefix_lens = [s[1] for s in train_ds._samples]
    print(f"  prefix len — min: {min(prefix_lens)}  max: {max(prefix_lens)}  "
          f"mean: {sum(prefix_lens)/len(prefix_lens):.1f}")

    # Target distribution (train)
    durations = [s[2] for s in train_ds._samples]
    print(f"  duration (sec) — min: {min(durations):.0f}  median: {np.median(durations):.0f}  "
          f"mean: {np.mean(durations):.0f}  max: {max(durations):.0f}")

    if device.type == "cuda":
        print("  Prefetching datasets to GPU …", end=" ", flush=True)
        train_ds.prefetch_to_device(device)
        val_ds.prefetch_to_device(device)
        print("done")

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True,  num_workers=args.workers)
    val_loader   = make_loader(val_ds,   args.batch_size, shuffle=False, num_workers=args.workers)

    model = StructuredModel(
        vocab_sizes        = vocab.vocab_sizes(),
        slot_names         = slot_names,
        numeric_slot_names = NUMERIC_SLOT_NAMES,
        d_model            = args.d_model,
        n_heads            = args.n_heads,
        n_layers           = args.n_layers,
        ffn_dim            = args.ffn_dim,
        dropout            = args.dropout,
        max_events         = args.max_events,
        n_outcomes         = 1,
    ).to(device)

    params = model.count_parameters()
    print(f"\nModel")
    print(f"  Parameters — total: {params['total']:,}  trainable: {params['trainable']:,}")
    print(f"  d_model={args.d_model}  n_heads={args.n_heads}  n_layers={args.n_layers}  ffn_dim={args.ffn_dim}")
    print(f"  max_events={args.max_events}  dropout={args.dropout}")

    if args.compile and device.type == "cuda":
        print("  torch.compile(): ON")
        model = torch.compile(model)
    else:
        print("  torch.compile(): OFF")

    criterion = RemainingTimeLoss(delta=args.huber_delta)
    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                      fused=(device.type == "cuda"))

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimiser, warmup_steps, total_steps)
    print(f"LR schedule: {warmup_steps} warmup / {total_steps} total steps")

    best_val_mae     = float("inf")
    best_val_loss    = float("inf")
    patience_counter = 0
    history          = []

    print(f"\n{'─'*70}")
    print(f"Training for up to {args.epochs} epochs (patience={args.patience})")
    print(f"Batch size:  {args.batch_size}")
    print(f"LR:          {args.lr}  wd={args.wd}  grad_clip={args.grad_clip}")
    print(f"Warmup:      {warmup_steps} steps ({args.warmup_ratio*100:.0f}% of {total_steps})")
    print(f"Huber delta: {args.huber_delta}")
    print(f"BF16:        {'ON' if use_bf16 else 'OFF'}")
    print(f"{'─'*70}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = run_epoch(model, train_loader, criterion, optimiser, scheduler, device,
                            training=True,  use_bf16=use_bf16, grad_clip=args.grad_clip,
                            desc=f"Epoch {epoch}/{args.epochs} train")
        val_m   = run_epoch(model, val_loader,   criterion, optimiser, scheduler, device,
                            training=False, use_bf16=use_bf16, grad_clip=args.grad_clip,
                            desc=f"Epoch {epoch}/{args.epochs} val  ")

        elapsed = time.time() - t0
        lr      = optimiser.param_groups[0]["lr"]
        vram_str = ""
        if device.type == "cuda":
            vram_str = f"  VRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB"
            torch.cuda.reset_peak_memory_stats()

        improved = "✓ best" if val_m["mae_sec"] < best_val_mae else f"  patience {patience_counter+1}/{args.patience}"
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {train_m['loss']:.4f} mae_hrs {train_m['mae_hrs']:.2f} | "
              f"val loss {val_m['loss']:.4f} mae_hrs {val_m['mae_hrs']:.2f} | "
              f"lr {lr:.2e} | {elapsed:.0f}s{vram_str} | {improved}")

        row = dict(epoch=epoch, train_loss=round(train_m["loss"],4),
                   train_mae_hrs=round(train_m["mae_hrs"],4), train_mae_sec=round(train_m["mae_sec"],1),
                   val_loss=round(val_m["loss"],4),
                   val_mae_hrs=round(val_m["mae_hrs"],4), val_mae_sec=round(val_m["mae_sec"],1),
                   lr=round(lr,7), secs=round(elapsed,1))
        history.append(row)

        if val_m["mae_sec"] < best_val_mae:
            best_val_mae, best_val_loss = val_m["mae_sec"], val_m["loss"]
            patience_counter = 0
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({
                "epoch": epoch, "model_state": raw.state_dict(),
                "val_mae_sec": best_val_mae, "val_loss": best_val_loss,
                "slot_names": slot_names,
                "numeric_slot_names": NUMERIC_SLOT_NAMES,
                "d_model": args.d_model, "n_heads": args.n_heads,
                "n_layers": args.n_layers, "ffn_dim": args.ffn_dim,
                "dropout": args.dropout, "max_events": args.max_events,
                "n_outcomes": 1,
            }, out_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'─'*70}")
    print(f"Best val MAE : {best_val_mae:.0f} sec  ({best_val_mae/3600:.2f} hrs)")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Epochs run   : {len(history)}")
    print(f"Checkpoint   : {out_dir}/best_model.pt")
    print(f"History      : {out_dir}/history.json")
    print(f"Vocab        : {out_dir}/vocab.json")
    print(f"\nTo evaluate:")
    print(f"  python evaluate.py --data {args.data} --checkpoint {out_dir}/best_model.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",        required=True,  help="Path to remaining_time.parquet")
    p.add_argument("--out_dir",     default="checkpoints_remaining_time")
    p.add_argument("--max_events",  type=int,   default=50)
    p.add_argument("--min_prefix",  type=int,   default=1)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=1024)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--wd",          type=float, default=0.01)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--warmup_ratio",type=float, default=0.10)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--d_model",     type=int,   default=512)
    p.add_argument("--n_heads",     type=int,   default=8)
    p.add_argument("--n_layers",    type=int,   default=6)
    p.add_argument("--ffn_dim",     type=int,   default=2048)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--bf16",        action="store_true", default=True)
    p.add_argument("--no_bf16",     dest="bf16", action="store_false")
    p.add_argument("--compile",     action="store_true", default=True)
    p.add_argument("--no_compile",  dest="compile", action="store_false")
    p.add_argument("--seed",        type=int,   default=42)
    main(p.parse_args())
