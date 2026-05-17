"""
model.py — Structured embedding model for cycle time prediction.

Architecture:
  Per-slot embedding tables (categorical) + linear projections (numeric)
  → concat → linear projection → transformer encoder → head

Numeric slots (Dur, Amt, FW, NT, MC, CS, OA) bypass embedding tables and are
projected from their log1p-transformed scalar values via nn.Linear(1, embed_dim).

Slot ablation is built-in: pass ablate_slots={"A", "R"} to zero out those embeddings
during forward, enabling clean feature importance measurement.

Optimised for RTX 4090:
  - PyTorch SDPA (Flash Attention equivalent, no external package)
  - Pre-LayerNorm transformer for stable high-LR training
  - BF16 compatible
"""
from __future__ import annotations
import torch
import torch.nn as nn


# Per-slot embedding dimensions (scales with vocabulary complexity)
SLOT_EMBED_DIMS: dict[str, int] = {
    "A":   64,   # activity  (~26 unique values)
    "R":   64,   # resource  (~100+ values)
    "Dur": 32,   # numeric: log1p(inter-event seconds)
    "WD":  16,   # 7 weekdays
    "LG":  16,   # loan goal
    "AT":  16,   # application type
    "Amt": 16,   # numeric: log1p(requested amount)
    "FW":  16,   # numeric: log1p(first withdrawal)
    "NT":  16,   # numeric: log1p(number of terms)
    "MC":  16,   # numeric: log1p(monthly cost)
    "CS":  16,   # numeric: log1p(credit score)
    "OA":  16,   # numeric: log1p(offered amount)
}


class StructuredModel(nn.Module):
    """
    Parameters
    ----------
    vocab_sizes       : {slot_name: vocabulary_size} for categorical slots only
    slot_names        : ordered list of all slot names (categorical + numeric)
    numeric_slot_names: ordered list of numeric slot names (subset of slot_names)
    d_model           : transformer hidden size
    n_heads           : attention heads
    n_layers          : transformer encoder layers
    ffn_dim           : feed-forward dimension (default 4× d_model)
    dropout           : dropout rate
    max_events        : maximum sequence length (for positional embeddings)
    n_outcomes        : unused (kept=1 for checkpoint compatibility)
    """

    def __init__(
        self,
        vocab_sizes:        dict[str, int],
        slot_names:         list,
        numeric_slot_names: list[str] = [],
        d_model:            int   = 256,
        n_heads:            int   = 8,
        n_layers:           int   = 4,
        ffn_dim:            int   = 1024,
        dropout:            float = 0.1,
        max_events:         int   = 50,
    ):
        super().__init__()
        self.slot_names          = slot_names
        self.d_model             = d_model
        self._numeric_slot_names = list(numeric_slot_names)
        self.numeric_slots       = frozenset(numeric_slot_names)

        self.embeddings = nn.ModuleDict({
            slot: nn.Embedding(
                vocab_sizes[slot],
                SLOT_EMBED_DIMS.get(slot, 8),
                padding_idx=0,
            )
            for slot in slot_names
            if slot not in self.numeric_slots and slot in vocab_sizes
        })

        self.numeric_projs = nn.ModuleDict({
            slot: nn.Linear(1, SLOT_EMBED_DIMS.get(slot, 8))
            for slot in numeric_slot_names
        })

        total_embed_dim = sum(SLOT_EMBED_DIMS.get(s, 8) for s in slot_names)

        self.input_proj = nn.Linear(total_embed_dim, d_model)
        self.pos_embed  = nn.Embedding(max_events + 1, d_model)  # +1 for CLS
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward = ffn_dim,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # pre-LN: stable with high LR
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for proj in self.numeric_projs.values():
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        slot_indices:   torch.Tensor,                    # (B, L, n_slots)
        padding_mask:   torch.Tensor,                    # (B, L) True=ignore
        numeric_values: torch.Tensor | None = None,     # (B, L, n_numeric)
        ablate_slots:   frozenset[str] = frozenset(),
    ) -> torch.Tensor:                                   # (B,) log1p(total_duration_sec)

        B, L, _ = slot_indices.shape

        embeds = []
        for s, slot in enumerate(self.slot_names):
            if slot in self.numeric_slots:
                ni   = self._numeric_slot_names.index(slot)
                vals = numeric_values[:, :, ni:ni+1]   # (B, L, 1)
                if slot in ablate_slots:
                    vals = vals[torch.randperm(B, device=vals.device)]
                e = self.numeric_projs[slot](vals)     # (B, L, embed_dim)
            else:
                idx = slot_indices[:, :, s]            # (B, L)
                if slot in ablate_slots:
                    idx = idx[torch.randperm(B, device=idx.device)]
                e = self.embeddings[slot](idx)         # (B, L, embed_dim)
            embeds.append(e)

        x = torch.cat(embeds, dim=-1)          # (B, L, total_embed_dim)
        x = self.input_proj(x)                 # (B, L, d_model)

        # Add positional embeddings (1-indexed; 0 reserved for CLS)
        positions = torch.arange(1, L + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_embed(positions)

        # Prepend CLS token (position 0)
        cls = self.cls_token.expand(B, -1, -1)                  # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                        # (B, L+1, d_model)

        # Extend mask: CLS is never masked
        cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, L+1)

        x = self.encoder(x, src_key_padding_mask=full_mask)
        return self.head(x[:, 0]).squeeze(-1)  # (B,) scalar per sample

    def count_parameters(self) -> dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


class RemainingTimeLoss(nn.Module):
    """
    Huber loss on log1p-transformed total duration.
    Reports both the raw Huber loss and MAE in original seconds via expm1.

    HuberLoss with delta=1.0 in log-space: transition at ~e-factor difference,
    robust to heavy-tailed case duration distributions.
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction="mean")

    def forward(
        self,
        preds:   torch.Tensor,   # (B,) model output in log1p-space
        targets: torch.Tensor,   # (B,) log1p(total_duration_sec), float32
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss = self.huber(preds.float(), targets.float())
        with torch.no_grad():
            pred_sec   = torch.expm1(preds.float().clamp(min=0))
            target_sec = torch.expm1(targets.float().clamp(min=0))
            mae_sec    = (pred_sec - target_sec).abs().mean().item()
        return loss, {"loss": loss.item(), "mae_sec": mae_sec, "mae_hrs": mae_sec / 3600.0}
