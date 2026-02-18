"""BERT-style masked modeling with joint classification.

Objectives:
- Masked token prediction (cross-entropy over token IDs).
- Masked value regression (MSE over intensities at masked positions).
- Optional classification from [CLS] token concurrently.

Inputs: docs_*.npz produced by the scenario builder with fields:
- parent_bins: array of parent ids
- tokens_idx: object array of 1D token ids or 2D pairs (frag, rt)
- tokens_val: object array of float values
- labels: array of labels (strings or ints) aligned to parent_bins
- kind: metadata (ignored here)
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from foundationmsms.logging.logger import get_logger


CLS_ID = 1
MASK_ID = 2
PAD_ID = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DocDataset(Dataset):
    def __init__(self, tokens_idx: Iterable, tokens_val: Iterable, labels: Iterable, pair_stride: Optional[int] = None):
        idx_clean: List[np.ndarray] = []
        val_clean: List[np.ndarray] = []
        label_clean: List[str] = []
        for idx, val, lab in zip(tokens_idx, tokens_val, labels):
            idx_arr = np.asarray(idx)
            val_arr = np.asarray(val)
            if idx_arr.size == 0:
                continue
            idx_clean.append(idx_arr)
            val_clean.append(val_arr)
            label_clean.append(str(lab))

        if not idx_clean:
            raise ValueError("No non-empty documents found in scenario file")

        self.is_pair = idx_clean[0].ndim == 2
        if self.is_pair:
            frag_max = max(int(x[:, 0].max()) for x in idx_clean)
            rt_max = max(int(x[:, 1].max()) for x in idx_clean)
            self.pair_stride = pair_stride or (rt_max + 1)
            self.base_vocab = int((frag_max * self.pair_stride) + rt_max + 1)
        else:
            self.pair_stride = None
            self.base_vocab = int(max(int(x.max()) for x in idx_clean) + 1)

        self.vocab = self.base_vocab + 3  # pad=0, cls=1, mask=2, tokens start at 3..
        self.idx_list = [np.asarray(x, dtype=np.int64) for x in idx_clean]
        self.val_list = [np.asarray(v, dtype=np.float32) for v in val_clean]

        # Label mapping to ints
        uniq = sorted(set(label_clean))
        self.label_to_id = {lab: i for i, lab in enumerate(uniq)}
        self.id_to_label = {i: lab for lab, i in self.label_to_id.items()}
        self.labels = np.array([self.label_to_id[l] for l in label_clean], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.idx_list)

    def __getitem__(self, i: int):
        idx = self.idx_list[i]
        val = self.val_list[i]
        if self.is_pair:
            frag = torch.as_tensor(idx[:, 0], dtype=torch.long)
            rt = torch.as_tensor(idx[:, 1], dtype=torch.long)
            idx_tensor = frag * self.pair_stride + rt
        else:
            idx_tensor = torch.as_tensor(idx, dtype=torch.long)
        val_tensor = torch.as_tensor(val, dtype=torch.float32)
        label = int(self.labels[i])
        return idx_tensor, val_tensor, label


def collate(batch):
    idxs, vals, labels = zip(*batch)
    max_len = max(x.shape[0] for x in idxs)
    padded_idx = torch.zeros(len(batch), max_len + 1, dtype=torch.long)  # +1 for CLS
    padded_val = torch.zeros(len(batch), max_len + 1, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len + 1, dtype=torch.bool)
    # place CLS at position 0
    padded_idx[:, 0] = CLS_ID
    mask[:, 0] = True
    for i, (idx, val) in enumerate(zip(idxs, vals)):
        length = idx.shape[0]
        padded_idx[i, 1 : length + 1] = idx + 3  # shift to leave room for specials
        padded_val[i, 1 : length + 1] = val
        mask[i, 1 : length + 1] = True
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    return padded_idx, padded_val, mask, labels_tensor


def create_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    # mask True where data exists; need attn mask of shape (B, 1, 1, L)
    return ~mask[:, None, None, :]


class ValueProjection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(1, dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.proj(values.unsqueeze(-1))


class BertLike(nn.Module):
    def __init__(self, vocab: int, num_classes: int, dim: int, heads: int, layers: int, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, dim, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(2048, dim)  # max length cap
        self.val_emb = ValueProjection(dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=int(dim * 4), dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.mlm_head = nn.Linear(dim, vocab)
        self.value_head = nn.Linear(dim, 1)
        self.cls_head = nn.Linear(dim, num_classes)

    def forward(self, idx: torch.Tensor, val: torch.Tensor, attn_mask: torch.Tensor):
        b, l = idx.shape
        pos_ids = torch.arange(l, device=idx.device).unsqueeze(0).expand(b, l)
        x = self.token_emb(idx) + self.pos_emb(pos_ids) + self.val_emb(val)
        encoded = self.encoder(x, src_key_padding_mask=attn_mask.squeeze(1).squeeze(1))
        mlm_logits = self.mlm_head(encoded)
        value_pred = self.value_head(encoded).squeeze(-1)
        cls_logits = self.cls_head(encoded[:, 0, :])
        return mlm_logits, value_pred, cls_logits


@dataclass
class TrainConfig:
    scenario: Path
    batch_size: int
    epochs: int
    lr: float
    mask_prob: float
    seed: int
    dim: int
    heads: int
    layers: int
    dropout: float
    device: Optional[str]


def mask_inputs(idx: torch.Tensor, val: torch.Tensor, mask: torch.Tensor, vocab: int, mask_prob: float):
    # idx already includes CLS (position 0) and shifted tokens
    B, L = idx.shape
    is_content = mask.clone()
    is_content[:, 0] = False  # do not mask CLS
    rand = torch.rand(B, L, device=idx.device)
    to_mask = (rand < mask_prob) & is_content
    # Targets
    target_tokens = idx.clone()
    target_values = val.clone()
    # Apply masking to inputs (idx) following BERT 80/10/10
    mask_rand = torch.rand(B, L, device=idx.device)
    # 80% -> [MASK]
    idx = torch.where(to_mask & (mask_rand < 0.8), torch.full_like(idx, MASK_ID), idx)
    # 10% -> random token
    random_tokens = torch.randint(3, vocab, (B, L), device=idx.device)
    idx = torch.where(to_mask & (mask_rand >= 0.8) & (mask_rand < 0.9), random_tokens, idx)
    # 10% -> keep original
    # Value targets only where masked
    value_mask = to_mask
    token_mask = to_mask
    return idx, val, token_mask, value_mask, target_tokens, target_values


def run_training(cfg: TrainConfig) -> None:
    logger = get_logger("bert_mlm_cls")
    set_seed(cfg.seed)

    sc = np.load(cfg.scenario, allow_pickle=True)
    parent_bins = sc["parent_bins"]
    tokens_idx = sc["tokens_idx"]
    tokens_val = sc["tokens_val"]
    labels = sc["labels"] if "labels" in sc else None
    if labels is None:
        raise ValueError("labels not found in scenario; rebuild with labels")

    dataset = DocDataset(tokens_idx, tokens_val, labels)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)

    num_classes = len(dataset.label_to_id)
    model = BertLike(vocab=dataset.vocab, num_classes=num_classes, dim=cfg.dim, heads=cfg.heads, layers=cfg.layers, dropout=cfg.dropout)

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ce_loss = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    mse_loss = nn.MSELoss()

    logger.info(
        "Config device=%s batch=%d lr=%.1e dim=%d heads=%d layers=%d mask_prob=%.2f classes=%d",
        device,
        cfg.batch_size,
        cfg.lr,
        cfg.dim,
        cfg.heads,
        cfg.layers,
        cfg.mask_prob,
        num_classes,
    )

    for epoch in range(cfg.epochs):
        model.train()
        total_mlm = 0.0
        total_val = 0.0
        total_cls = 0.0
        steps = 0
        for batch in dataloader:
            idx, val, mask, labels_tensor = batch
            idx = idx.to(device)
            val = val.to(device)
            mask = mask.to(device)
            labels_tensor = labels_tensor.to(device)

            idx_masked, val_masked, token_mask, value_mask, target_tokens, target_values = mask_inputs(idx, val, mask, dataset.vocab, cfg.mask_prob)

            attn_mask = create_padding_mask(mask)
            mlm_logits, val_pred, cls_logits = model(idx_masked, val_masked, attn_mask)

            # MLM loss
            if token_mask.any():
                mlm_loss = ce_loss(mlm_logits[token_mask], target_tokens[token_mask])
            else:
                mlm_loss = torch.tensor(0.0, device=device)

            # Value regression loss
            if value_mask.any():
                val_loss = mse_loss(val_pred[value_mask], target_values[value_mask])
            else:
                val_loss = torch.tensor(0.0, device=device)

            cls_loss = ce_loss(cls_logits, labels_tensor)

            loss = mlm_loss + val_loss + cls_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_mlm += mlm_loss.item()
            total_val += val_loss.item()
            total_cls += cls_loss.item()
            steps += 1

        if steps:
            logger.info(
                "epoch=%d mlm=%.4f val=%.4f cls=%.4f",
                epoch,
                total_mlm / steps,
                total_val / steps,
                total_cls / steps,
            )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BERT MLM+CLS trainer")
    p.add_argument("--scenario", type=Path, required=True, help="Path to docs_*.npz with labels")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", type=str, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = TrainConfig(
        scenario=args.scenario,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        mask_prob=args.mask_prob,
        seed=args.seed,
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        device=args.device,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
