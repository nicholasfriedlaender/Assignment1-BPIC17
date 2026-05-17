# Assignment1 BPIC17 — Code

Analysis of the **BPIC17** event log (Dutch financial institution loan application process) using process mining and deep learning.


## Notebooks

### part_3.2.ipynb — Event log statistics (§3.2)
Computes basic descriptive statistics from the raw log: case/event/variant counts, case length and duration distributions, and two custom process metrics (fraction of cases with >1 offer; loan grant rate via A_Pending).

### part_3.3.ipynb — Discovery, conformance & decision mining (§3.3)
1. **Filtering pipeline** — produces log variants from the raw XES (complete events only → outcome-labelled → top-variant subsets).
2. **Process model discovery** — Inductive Miner at noise thresholds 0.2/0.3/0.4 and Heuristics Miner, applied to both the full outcome log and a top-variant subset.
3. **Conformance checking** — token-based replay measuring fitness, precision, generalization, simplicity, size, and CFC for all six models.
4. **Decision mining** — depth-1 decision trees at six process gateways to explain routing decisions (e.g. CreditScore at GW4/GW5, LoanGoal at GW2).
5. Exports all Petri net and BPMN figures to `report/figures/`.

### part_3.4 — Remaining-time prediction

A structured-embedding Transformer that predicts total case cycle time from a prefix of events.

#### Architecture (`model.py`)

- Each event slot gets its own embedding: categorical slots use `nn.Embedding`, numeric slots use `nn.Linear(1, embed_dim)` on their log1p-transformed value.
- All slot embeddings are concatenated and projected to `d_model`.
- A **pre-LayerNorm Transformer encoder** with a CLS token reads the event sequence.
- The CLS output is passed through a regression head predicting `log1p(total_duration_sec)`.
- Loss: **Huber loss in log-space** (robust to heavy-tailed case durations).

#### Usage

**1. Install dependencies**

```bash
pip install -r part_3.4/requirements.txt
```

**2. Preprocess** (converts CSV → case-level parquet)

```bash
cd part_3.4
python preprocess.py logs_filtered.csv data/remaining_time.parquet
```

**3. Train**

```bash
python train.py --data data/remaining_time.parquet
```

Checkpoints are saved to `checkpoints_remaining_time/` whenever validation MAE improves.

**4. Evaluate**

```bash
python evaluate.py \
    --data data/remaining_time.parquet \
    --checkpoint checkpoints_remaining_time/best_model.pt
```

Outputs overall MAE / RMSE / MAPE, MAE broken down by prefix length, and slot ablation importance (permutation-based, measures MAE increase when each slot is shuffled).

**5. Smoke test**

```bash
python smoke_test.py
```

## Notebook dependencies

```bash
pip install pipenv
pipenv install   # reads Pipfile
pipenv run jupyter notebook
```
