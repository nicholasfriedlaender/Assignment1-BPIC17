"""
preprocess.py — BPIC2017.csv → remaining_time.parquet

One row per CASE. Each slot column holds pipe-separated per-event values.
total_duration_sec is a single float: (case_end_ts - case_start_ts) in seconds.
Generating prefixes is deferred to the dataset (avoids storing redundant data).

Usage:
    python preprocess.py ../BPIC2017.csv data/remaining_time.parquet [--max-events 50]
"""
from __future__ import annotations
import sys, argparse, math
from pathlib import Path
import pandas as pd

for _p in [str(Path(__file__).parent), str(Path(__file__).parent.parent)]:
    if _p not in sys.path:
        sys.path.append(_p)
from schema import EVENT_SLOTS, OUTCOME_LABELS

SLOT_NAMES    = [s[0] for s in EVENT_SLOTS]
_TERMINAL     = set(OUTCOME_LABELS)
_NUMERIC_KEYS = {"Dur", "Amt", "FW", "NT", "MC", "CS", "OA"}


def _log1p_num(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "0.0"
    return str(math.log1p(max(0.0, float(v))))


def preprocess(input_csv: Path, output_path: Path, max_events: int = 50) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], format="mixed", utc=True)
    df = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
    df["prev_ts"]      = df.groupby("case:concept:name")["time:timestamp"].shift(1)
    df["duration_sec"] = (df["time:timestamp"] - df["prev_ts"]).dt.total_seconds()

    # Vectorised case-level total duration
    case_ts   = df.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
    case_dur  = (case_ts["max"] - case_ts["min"]).dt.total_seconds().rename("total_duration_sec")

    _nan = pd.Series([float("nan")] * len(df), index=df.index, dtype=object)
    slot_cols = {
        key: (df[col] if col in df.columns else _nan).map(_log1p_num if key in _NUMERIC_KEYS else fn)
        for key, fn, col in EVENT_SLOTS
    }

    records = []
    for case_id, grp in df.groupby("case:concept:name", sort=False):
        if len(grp) < 2:
            continue
        acts               = grp["concept:name"].tolist()
        total_duration_sec = float(case_dur.get(case_id, 0.0))

        slot_seqs: dict[str, list] = {k: [] for k in SLOT_NAMES}
        for i, idx in enumerate(grp.index):
            if i >= max_events:
                break
            if acts[i] in _TERMINAL:
                break
            for k in SLOT_NAMES:
                v = slot_cols[k][idx]
                slot_seqs[k].append("NaN" if (v is None or str(v) in ("nan", "None", "NaN")) else str(v))

        if not slot_seqs[SLOT_NAMES[0]]:
            continue

        row = {"case_id": case_id, "total_duration_sec": total_duration_sec,
               "n_events": len(slot_seqs[SLOT_NAMES[0]])}
        for k in SLOT_NAMES:
            row[k] = "|".join(slot_seqs[k])
        records.append(row)

    out = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if str(output_path).endswith(".parquet"):
        out.to_parquet(output_path, index=False)
    else:
        out.to_csv(output_path, index=False)

    dur = out["total_duration_sec"]
    print(f"Cases:     {len(out):,}")
    print(f"Duration (sec) — min: {dur.min():.0f}  median: {dur.median():.0f}  "
          f"mean: {dur.mean():.0f}  max: {dur.max():.0f}")
    print(f"Saved  →   {output_path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input",         type=Path)
    p.add_argument("output",        type=Path)
    p.add_argument("--max-events",  type=int, default=50)
    args = p.parse_args()
    preprocess(args.input, args.output, args.max_events)
