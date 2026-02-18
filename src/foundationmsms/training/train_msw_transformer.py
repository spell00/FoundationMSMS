"""Quick trainer for the MSWTransformer on prebuilt document scenarios.

Loads a docs_*.npz scenario (from the notebook generator), reports dataset
statistics, and runs a masked MSE reconstruction loop. Use this to sanity-check
sequence shapes and how the transformer processes inputs.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..logging.logger import get_logger
from ..models import MSWConfig, MSWTransformer


# ---------------------
# Dataset and collation
# ---------------------


class DocDataset(Dataset):
    """Document dataset over tokens_idx/tokens_val arrays.

    Supports 1D token ids or 2D (frag, rt) pairs by hashing pairs into a single
    vocabulary index. Optionally carries labels for classification.
    """

    def __init__(self, tokens_idx: Iterable, tokens_val: Iterable, labels: Optional[Iterable] = None, pair_stride: Optional[int] = None):
        idx_clean: List[np.ndarray] = []
        val_clean: List[np.ndarray] = []
        label_clean: List[str] = []
        has_labels = labels is not None
        if has_labels:
            for idx, val, lab in zip(tokens_idx, tokens_val, labels):
                idx_arr = np.asarray(idx)
                val_arr = np.asarray(val)
                if idx_arr.size == 0:
                    continue
                idx_clean.append(idx_arr)
                val_clean.append(val_arr)
                label_clean.append(str(lab))
        else:
            for idx, val in zip(tokens_idx, tokens_val):
                idx_arr = np.asarray(idx)
                val_arr = np.asarray(val)
                if idx_arr.size == 0:
                    continue
                idx_clean.append(idx_arr)
                val_clean.append(val_arr)

        if not idx_clean:
            raise ValueError("No non-empty documents found in scenario file")

        self.is_pair = idx_clean[0].ndim == 2
        if self.is_pair:
            if any(x.ndim != 2 or x.shape[1] != 2 for x in idx_clean):
                raise ValueError("All pair documents must have shape (L, 2)")
            frag_max = max(int(x[:, 0].max()) for x in idx_clean)
            rt_max = max(int(x[:, 1].max()) for x in idx_clean)
            self.pair_stride = pair_stride or (rt_max + 1)
            self.vocab = int((frag_max * self.pair_stride) + rt_max + 1)
        else:
            if any(x.ndim != 1 for x in idx_clean):
                raise ValueError("All single-token documents must be 1D")
            self.pair_stride = None
            self.vocab = int(max(int(x.max()) for x in idx_clean) + 1)

        self.idx_list = [np.asarray(x, dtype=np.int64) for x in idx_clean]
        self.val_list = [np.asarray(v, dtype=np.float32) for v in val_clean]

        if has_labels:
            uniq = sorted(set(label_clean))
            self.label_to_id = {lab: i for i, lab in enumerate(uniq)}
            self.id_to_label = {i: lab for lab, i in self.label_to_id.items()}
            self.labels = np.array([self.label_to_id[l] for l in label_clean], dtype=np.int64)
        else:
            self.label_to_id = None
            self.id_to_label = None
            self.labels = None

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
        if self.labels is None:
            return idx_tensor, val_tensor
        label = int(self.labels[i])
        return idx_tensor, val_tensor, label


def collate_recon(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
    idxs, vals = zip(*batch)
    max_len = max(x.shape[0] for x in idxs)
    padded_idx = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_val = torch.zeros(len(batch), max_len, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (idx, val) in enumerate(zip(idxs, vals)):
        length = idx.shape[0]
        padded_idx[i, :length] = idx
        padded_val[i, :length] = val
        mask[i, :length] = True
    return padded_idx, padded_val, mask


def collate_clf(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, int]], vocab: int):
    idxs, vals, labels = zip(*batch)
    max_len = max(x.shape[0] for x in idxs)
    cls_id = vocab + 1  # pad=0, tokens in [0,vocab-1], cls at vocab+1
    padded_idx = torch.zeros(len(batch), max_len + 1, dtype=torch.long)
    padded_val = torch.zeros(len(batch), max_len + 1, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len + 1, dtype=torch.bool)
    padded_idx[:, 0] = cls_id
    mask[:, 0] = True
    for i, (idx, val) in enumerate(zip(idxs, vals)):
        length = idx.shape[0]
        padded_idx[i, 1 : length + 1] = idx + 1  # shift tokens by +1 to avoid pad collision
        padded_val[i, 1 : length + 1] = val
        mask[i, 1 : length + 1] = True
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    return padded_idx, padded_val, mask, labels_tensor, cls_id


# -------------
# Descriptions
# -------------


def describe_voxel_dir(voxel_dir: Path, npz_limit: Optional[int], logger) -> None:
    npz_files = sorted(voxel_dir.glob("*.npz"))
    if npz_limit is not None:
        npz_files = npz_files[:npz_limit]
    if not npz_files:
        logger.info("No voxel files found in %s", voxel_dir)
        return

    doc_counts: List[int] = []
    for p in npz_files:
        npz = np.load(p)
        coords = npz["coords"]
        if coords.size == 0:
            continue
        parents = np.unique(coords[:, 0])
        doc_counts.append(int(parents.size))

    if doc_counts:
        logger.info(
            "Docs per voxel file: files=%d avg=%.1f min=%d max=%d",
            len(npz_files),
            statistics.mean(doc_counts),
            min(doc_counts),
            max(doc_counts),
        )
    else:
        logger.info("Voxel files contained no parent bins")


def describe_docs(tokens_idx: Iterable, logger) -> List[int]:
    lengths = [int(len(np.asarray(x))) for x in tokens_idx]
    if not lengths:
        logger.info("No documents to summarize")
        return []

    def pct(p: float) -> float:
        return float(np.percentile(lengths, p))

    logger.info(
        "Documents=%d tokens/doc avg=%.1f min=%d p50=%.1f p90=%.1f p99=%.1f max=%d",
        len(lengths),
        statistics.mean(lengths),
        min(lengths),
        pct(50),
        pct(90),
        pct(99),
        max(lengths),
    )
    return lengths


# ---------
# Training
# ---------


@dataclass
class TrainConfig:
    scenario_path: Path
    voxel_dir: Optional[Path]
    npz_limit: Optional[int]
    batch_size: int
    epochs: int
    lr: float
    dim: int
    heads: int
    windows: Tuple[int, ...]
    device: Optional[str]
    task: str
    recon_weight: float
    clf_weight: float
    test_frac: float
    cv_folds: int
    seed: int
    early_stop_patience: int
    early_stop_min_delta: float


def log_sample_flow(logger, idx: torch.Tensor, mask: torch.Tensor, tok: torch.Tensor, out: torch.Tensor, pred: torch.Tensor) -> None:
    logger.info(
        "Batch shapes idx=%s mask=%s embed=%s msw=%s head=%s",
        tuple(idx.shape),
        tuple(mask.shape),
        tuple(tok.shape),
        tuple(out.shape),
        tuple(pred.shape),
    )


def run_training(cfg: TrainConfig, args) -> None:
    logger = get_logger("msw_train")

    if cfg.voxel_dir:
        describe_voxel_dir(cfg.voxel_dir, cfg.npz_limit, logger)

    sc = np.load(cfg.scenario_path, allow_pickle=True)
    parent_bins = sc["parent_bins"]
    tokens_idx = sc["tokens_idx"]
    tokens_val = sc["tokens_val"]
    labels = sc["labels"] if "labels" in sc else None
    kind = sc.get("kind", None)

    logger.info("Loaded scenario %s kind=%s docs=%d", cfg.scenario_path, kind, len(parent_bins))
    lengths = describe_docs(tokens_idx, logger)

    if cfg.task in {"clf", "joint"} and labels is None:
        raise ValueError("Classification task requires 'labels' in scenario NPZ")


    # Assume labels is a list of tuples: (dataset_id, label) or similar
    # If not, fallback to single dataset
    dataset_ids = set()
    sample_to_dataset = []
    if labels is not None and hasattr(labels[0], '__iter__') and not isinstance(labels[0], str):
        # e.g. [(PXD012353, 0), (PXD028735, 1), ...]
        for lab in labels:
            dataset_ids.add(lab[0])
            sample_to_dataset.append(lab[0])
        label_values = [lab[1] for lab in labels]
    else:
        dataset_ids = {"default"}
        sample_to_dataset = ["default"] * len(labels) if labels is not None else []
        label_values = labels

    dataset = DocDataset(tokens_idx, tokens_val, labels=label_values)
    logger.info(
        "Vocab size=%d is_pair=%s pair_stride=%s datasets=%s",
        dataset.vocab,
        dataset.is_pair,
        dataset.pair_stride if dataset.is_pair else "-",
        list(dataset_ids),
    )

    collate_fn = (lambda b: collate_clf(b, dataset.vocab)) if cfg.task in {"clf", "joint"} else collate_recon

    rng = np.random.default_rng(cfg.seed)
    all_idx = np.arange(len(dataset))
    rng.shuffle(all_idx)
    test_size = int(len(dataset) * cfg.test_frac) if cfg.test_frac > 0 else 0
    test_size = min(max(test_size, 0), len(dataset) - 1) if len(dataset) > 1 else 0
    test_idx = all_idx[:test_size]
    remain_idx = all_idx[test_size:]
    if cfg.cv_folds > 1 and len(remain_idx) < cfg.cv_folds:
        raise ValueError("Not enough samples for requested cv_folds")
    if cfg.cv_folds > 1 and any(len(x) == 0 for x in np.array_split(remain_idx, cfg.cv_folds)):
        raise ValueError("Empty fold encountered; reduce cv_folds or dataset size")

    folds = np.array_split(remain_idx, cfg.cv_folds) if cfg.cv_folds > 1 else [remain_idx]

    def make_loader(indices: Optional[np.ndarray], shuffle: bool) -> Optional[DataLoader]:
        if indices is None or len(indices) == 0:
            return None
        subset = torch.utils.data.Subset(dataset, indices.tolist())
        return DataLoader(subset, batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate_fn)

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def init_models():
        model_cfg = MSWConfig(dim=cfg.dim, num_heads=cfg.heads, window_sizes=cfg.windows)
        model = MSWTransformer(model_cfg)
        use_cls = cfg.task in {"clf", "joint"}
        embed = nn.Embedding(dataset.vocab + (2 if use_cls else 1), model_cfg.dim, padding_idx=0)
        head_recon: Optional[nn.Linear] = None
        head_clf_dict = {}
        if cfg.task in {"recon", "joint"}:
            head_recon = nn.Linear(model_cfg.dim, 1)
        if cfg.task in {"clf", "joint"}:
            for dsid in dataset_ids:
                # Only create head if dataset has labels
                if any(d == dsid for d in sample_to_dataset):
                    n_classes = len(set(l for d, l in zip(sample_to_dataset, label_values) if d == dsid))
                    head_clf_dict[dsid] = nn.Linear(model_cfg.dim, n_classes)
        # If no labeled datasets, head_clf_dict stays empty
        model.to(device)
        embed.to(device)
        if head_recon is not None:
            head_recon.to(device)
        for head in head_clf_dict.values():
            head.to(device)
        params = list(model.parameters()) + list(embed.parameters())
        if head_recon is not None:
            params += list(head_recon.parameters())
        for head in head_clf_dict.values():
            params += list(head.parameters())
        opt = torch.optim.Adam(params, lr=cfg.lr)
        ce_loss = nn.CrossEntropyLoss()
        return model, embed, head_recon, head_clf_dict, opt, ce_loss

    def evaluate(loader: Optional[DataLoader], model, embed, head_recon, head_clf, ce_loss):
        if loader is None:
            return None
        model.eval()
        embed.eval()
        if head_recon is not None:
            head_recon.eval()
        if head_clf is not None:
            head_clf.eval()
        recon_loss_sum = 0.0
        clf_loss_sum = {dsid: 0.0 for dsid in head_clf.keys()}
        clf_correct = {dsid: 0 for dsid in head_clf.keys()}
        clf_total = {dsid: 0 for dsid in head_clf.keys()}
        batch_count = 0
        with torch.no_grad():
            for batch in loader:
                if cfg.task in {"clf", "joint"}:
                    idx, val, mask, labels_tensor, cls_id = batch
                    labels_tensor = labels_tensor.to(device)
                    # Assume batch is sorted/grouped by dataset, or add dataset_id info to batch
                    # For now, assume all samples in batch are from the same dataset
                    batch_dataset = sample_to_dataset[batch_count] if sample_to_dataset else "default"
                else:
                    idx, val, mask = batch
                    labels_tensor = None
                    batch_dataset = None

                idx = idx.to(device)
                val = val.to(device)
                mask = mask.to(device)

                tok = embed(idx)
                out = model(tok)

                if cfg.task in {"recon", "joint"} and head_recon is not None:
                    pred_recon = head_recon(out).squeeze(-1)
                    recon_mask = mask
                    if cfg.task == "joint":
                        recon_mask = recon_mask.clone()
                        recon_mask[:, 0] = False
                    masked_pred = pred_recon[recon_mask]
                    masked_val = val[recon_mask]
                    recon_loss_sum += torch.mean((masked_pred - masked_val) ** 2).item()

                if cfg.task in {"clf", "joint"} and head_clf and ce_loss is not None and labels_tensor is not None and batch_dataset in head_clf:
                    pred_clf = head_clf[batch_dataset](out[:, 0, :])
                    clf_loss_sum[batch_dataset] += ce_loss(pred_clf, labels_tensor).item()
                    clf_correct[batch_dataset] += (pred_clf.argmax(dim=-1) == labels_tensor).sum().item()
                    clf_total[batch_dataset] += labels_tensor.numel()

                batch_count += 1

        if batch_count == 0:
            return None
        result = {"batches": batch_count}
        if cfg.task in {"recon", "joint"}:
            result["recon_loss"] = recon_loss_sum / batch_count
        if cfg.task in {"clf", "joint"}:
            result["clf_loss"] = {dsid: clf_loss_sum[dsid] / max(1, batch_count) for dsid in clf_loss_sum}
            result["acc"] = {dsid: clf_correct[dsid] / max(1, clf_total[dsid]) for dsid in clf_correct}
        return result

    def score_for_early_stop(metrics: dict) -> Optional[float]:
        if metrics is None:
            return None
        if cfg.task == "recon":
            return metrics.get("recon_loss")
        if cfg.task == "clf":
            return metrics.get("clf_loss")
        # joint: combine losses using same weights
        recon = metrics.get("recon_loss")
        clf = metrics.get("clf_loss")
        if recon is None or clf is None:
            return None
        return (cfg.recon_weight * recon) + (cfg.clf_weight * clf)

    # --- Model and optimizer initialization ---
    model, embed, head_recon, head_clf, opt, ce_loss = init_models()
    # --- WARMUP PHASE ---
    if args.warmup_epochs > 0:
        logger.info(f"Starting warmup phase: {args.warmup_epochs} epochs (unsupervised reconstruction only)")
        # If transductive_inference, include test samples (without labels) in warmup
        if args.transductive_inference:
            # Create a dataset with all train+test indices, but labels=None for test
            all_indices = np.concatenate([remain_idx, test_idx])
            # For test samples, set label to None
            if labels is not None:
                labels_full = list(labels)
                for i in range(len(labels_full)):
                    if i in test_idx:
                        labels_full[i] = None
                dataset_warmup = DocDataset(tokens_idx, tokens_val, labels=labels_full)
            else:
                dataset_warmup = DocDataset(tokens_idx, tokens_val, labels=None)
            indices_warmup = all_indices
        else:
            dataset_warmup = dataset
            indices_warmup = remain_idx
        collate_fn_warmup = collate_recon
        train_loader_warmup = DataLoader(torch.utils.data.Subset(dataset_warmup, indices_warmup.tolist()), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn_warmup)
        model.train()
        embed.train()
        if head_recon is not None:
            head_recon.train()
        for epoch in range(args.warmup_epochs):
            recon_loss_sum = 0.0
            batch_count = 0
            for idx, val, mask in train_loader_warmup:
                idx = idx.to(device)
                val = val.to(device)
                mask = mask.to(device)
                tok = embed(idx)
                out = model(tok)
                pred_recon = head_recon(out).squeeze(-1)
                masked_pred = pred_recon[mask]
                masked_val = val[mask]
                recon_loss = torch.mean((masked_pred - masked_val) ** 2)
                opt.zero_grad()
                recon_loss.backward()
                opt.step()
                recon_loss_sum += recon_loss.item()
                batch_count += 1
            if batch_count:
                logger.info(f"[warmup] epoch={epoch+1}/{args.warmup_epochs} recon_loss={recon_loss_sum/batch_count:.4f} batches={batch_count}")
        # Save model state after warmup
        torch.save({
            'model': model.state_dict(),
            'embed': embed.state_dict(),
            'head_recon': head_recon.state_dict() if head_recon is not None else None,
        }, args.warmup_out)
        logger.info(f"Warmup phase complete. Model saved to {args.warmup_out}")
    # --- Main training loop continues below ---
    num_folds = cfg.cv_folds if cfg.cv_folds > 1 else 1
    for fold_idx in range(num_folds):
        # Reload model from warmup.pth before main training
        logger.info(f"Reloading model from {args.warmup_out} for main training phase")
        state = torch.load(args.warmup_out, map_location=device)
        model.load_state_dict(state['model'])
        embed.load_state_dict(state['embed'])
        if head_recon is not None and state.get('head_recon') is not None:
            head_recon.load_state_dict(state['head_recon'])

        val_idx = folds[fold_idx] if cfg.cv_folds > 1 else None
        train_idx = np.concatenate([folds[i] for i in range(len(folds)) if i != fold_idx]) if cfg.cv_folds > 1 else remain_idx

        if train_idx.size == 0:
            raise ValueError("Training split is empty")

        train_loader = make_loader(train_idx, shuffle=True)
        val_loader = make_loader(val_idx, shuffle=False)
        test_loader = make_loader(test_idx, shuffle=False)

        logger.info(
            "Fold %d/%d train=%d val=%d test=%d",
            fold_idx + 1,
            num_folds,
            len(train_idx),
            len(val_idx) if val_idx is not None else 0,
            len(test_idx),
        )


        logger.info(
            "Config task=%s device=%s batch=%d lr=%.1e dim=%d heads=%d windows=%s epochs=%d",
            cfg.task,
            device,
            cfg.batch_size,
            cfg.lr,
            cfg.dim,
            cfg.heads,
            cfg.windows,
            cfg.epochs,
        )



        step = 0
        model.train()
        embed.train()
        best_score = float("inf")
        best_states = None
        epochs_no_improve = 0
        for epoch in range(cfg.epochs):
            recon_loss_sum = 0.0
            clf_loss_sum = 0.0
            clf_correct = 0
            clf_total = 0
            batch_count = 0
            for batch in train_loader:
                if cfg.task in {"clf", "joint"}:
                    idx, val, mask, labels_tensor, cls_id = batch
                    labels_tensor = labels_tensor.to(device)
                else:
                    idx, val, mask = batch
                    labels_tensor = None

                idx = idx.to(device)
                val = val.to(device)
                mask = mask.to(device)

                tok = embed(idx)
                out = model(tok)

                recon_loss = None
                clf_loss = None
                pred_recon = None
                pred_clf = None

                if cfg.task in {"recon", "joint"} and head_recon is not None:
                    pred_recon = head_recon(out).squeeze(-1)
                    recon_mask = mask.clone()
                    if cfg.task == "joint":
                        recon_mask[:, 0] = False  # exclude CLS from reconstruction loss
                    masked_pred = pred_recon[recon_mask]
                    masked_val = val[recon_mask]
                    recon_loss = torch.mean((masked_pred - masked_val) ** 2)
                    recon_loss_sum += recon_loss.item()

                if cfg.task in {"clf", "joint"} and head_clf is not None and ce_loss is not None and labels_tensor is not None:
                    pred_clf = head_clf(out[:, 0, :])  # CLS at position 0
                    clf_loss = ce_loss(pred_clf, labels_tensor)
                    clf_loss_sum += clf_loss.item()
                    with torch.no_grad():
                        clf_correct += (pred_clf.argmax(dim=-1) == labels_tensor).sum().item()
                        clf_total += labels_tensor.numel()

                if cfg.task == "joint":
                    if recon_loss is None or clf_loss is None:
                        raise RuntimeError("Joint task requires both losses to be computed")
                    loss = (cfg.recon_weight * recon_loss) + (cfg.clf_weight * clf_loss)
                elif cfg.task == "clf":
                    if clf_loss is None:
                        raise RuntimeError("Classification loss missing")
                    loss = clf_loss
                else:
                    if recon_loss is None:
                        raise RuntimeError("Reconstruction loss missing")
                    loss = recon_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                step += 1
                batch_count += 1
                if step == 1:
                    sample_pred = pred_clf if cfg.task in {"clf", "joint"} else pred_recon
                    log_sample_flow(logger, idx, mask, tok, out, sample_pred)

            if batch_count:
                if cfg.task == "clf":
                    acc = clf_correct / max(1, clf_total)
                    logger.info("fold=%d epoch=%d clf_loss=%.4f acc=%.4f batches=%d", fold_idx + 1, epoch, clf_loss_sum / batch_count, acc, batch_count)
                elif cfg.task == "joint":
                    acc = clf_correct / max(1, clf_total)
                    logger.info(
                        "fold=%d epoch=%d recon_loss=%.4f clf_loss=%.4f acc=%.4f batches=%d",
                        fold_idx + 1,
                        epoch,
                        recon_loss_sum / batch_count,
                        clf_loss_sum / batch_count,
                        acc,
                        batch_count,
                    )
                else:
                    logger.info("fold=%d epoch=%d recon_loss=%.4f batches=%d", fold_idx + 1, epoch, recon_loss_sum / batch_count, batch_count)

            if val_loader is not None:
                val_metrics = evaluate(val_loader, model, embed, head_recon, head_clf, ce_loss)
                if val_metrics:
                    val_score = score_for_early_stop(val_metrics)
                    if cfg.task == "recon":
                        logger.info("fold=%d epoch=%d val_recon_loss=%.4f batches=%d", fold_idx + 1, epoch, val_metrics.get("recon_loss", 0.0), val_metrics["batches"])
                    elif cfg.task == "clf":
                        logger.info(
                            "fold=%d epoch=%d val_clf_loss=%.4f val_acc=%.4f batches=%d",
                            fold_idx + 1,
                            epoch,
                            val_metrics.get("clf_loss", 0.0),
                            val_metrics.get("acc", 0.0),
                            val_metrics["batches"],
                        )
                    else:
                        logger.info(
                            "fold=%d epoch=%d val_recon_loss=%.4f val_clf_loss=%.4f val_acc=%.4f batches=%d",
                            fold_idx + 1,
                            epoch,
                            val_metrics.get("recon_loss", 0.0),
                            val_metrics.get("clf_loss", 0.0),
                            val_metrics.get("acc", 0.0),
                            val_metrics["batches"],
                        )

                    if val_score is not None:
                        if val_score + cfg.early_stop_min_delta < best_score:
                            best_score = val_score
                            epochs_no_improve = 0
                            best_states = (
                                copy.deepcopy(model.state_dict()),
                                copy.deepcopy(embed.state_dict()),
                                copy.deepcopy(head_recon.state_dict()) if head_recon is not None else None,
                                copy.deepcopy(head_clf.state_dict()) if head_clf is not None else None,
                                copy.deepcopy(opt.state_dict()),
                            )
                        else:
                            epochs_no_improve += 1

                        if cfg.early_stop_patience > 0 and epochs_no_improve >= cfg.early_stop_patience:
                            logger.info(
                                "fold=%d early stopping at epoch=%d best_score=%.4f",
                                fold_idx + 1,
                                epoch,
                                best_score,
                            )
                            break

            model.train()
            embed.train()
            if head_recon is not None:
                head_recon.train()
            if head_clf is not None:
                head_clf.train()

        # restore best states if we early-stopped or simply to use best val model
        if best_states is not None and val_loader is not None:
            model_state, embed_state, head_recon_state, head_clf_state, opt_state = best_states
            model.load_state_dict(model_state)
            embed.load_state_dict(embed_state)
            if head_recon is not None and head_recon_state is not None:
                head_recon.load_state_dict(head_recon_state)
            if head_clf is not None and head_clf_state is not None:
                head_clf.load_state_dict(head_clf_state)
            opt.load_state_dict(opt_state)

        test_metrics = evaluate(test_loader, model, embed, head_recon, head_clf, ce_loss)
        if test_metrics:
            if cfg.task == "recon":
                logger.info("fold=%d test_recon_loss=%.4f batches=%d", fold_idx + 1, test_metrics.get("recon_loss", 0.0), test_metrics["batches"])
            elif cfg.task == "clf":
                logger.info(
                    "fold=%d test_clf_loss=%.4f test_acc=%.4f batches=%d",
                    fold_idx + 1,
                    test_metrics.get("clf_loss", 0.0),
                    test_metrics.get("acc", 0.0),
                    test_metrics["batches"],
                )
            else:
                logger.info(
                    "fold=%d test_recon_loss=%.4f test_clf_loss=%.4f test_acc=%.4f batches=%d",
                    fold_idx + 1,
                    test_metrics.get("recon_loss", 0.0),
                    test_metrics.get("clf_loss", 0.0),
                    test_metrics.get("acc", 0.0),
                    test_metrics["batches"],
                )

    logger.info("Training complete")


# ---------
# CLI
# ---------


def parse_windows(value: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("windows must be a comma-separated list of ints")
    return tuple(int(p) for p in parts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MSWTransformer on document scenarios")
    parser.add_argument("--transductive-inference", action="store_true", help="Allow model to see test files during training (transductive inference)")
    parser.add_argument("--scenario", type=Path, required=True, help="Path to docs_*.npz scenario file")
    parser.add_argument("--voxel-dir", type=Path, help="Optional voxel dir to report docs/file stats")
    parser.add_argument("--npz-limit", type=int, default=None, help="Limit voxel files for reporting only")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=128, help="Model embedding dim")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--windows", type=parse_windows, default=parse_windows("5,10,20"), help="Comma-separated window sizes")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu or cuda)")
    parser.add_argument("--task", type=str, choices=["recon", "clf", "joint"], default="joint", help="recon = masked MSE, clf = CLS classification, joint = both")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Number of unsupervised warmup epochs (reconstruction only, before main training)")
    parser.add_argument("--warmup-out", type=str, default="warmup.pth", help="Path to save model after warmup phase")
    parser.add_argument("--recon-weight", type=float, default=1.0, help="Weight for reconstruction loss when joint task is used")
    parser.add_argument("--clf-weight", type=float, default=1.0, help="Weight for classification loss when joint task is used")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction of data to reserve for test set")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds (1 disables CV)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Epochs with no val improvement before stopping (0 disables)")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Minimum improvement in monitored val metric to reset patience")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of unsupervised warmup epochs (reconstruction only, before main training)")
    parser.add_argument("--warmup-out", type=str, default="warmup.pth", help="Path to save model after warmup phase")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    npz_limit = None if (args.npz_limit is not None and args.npz_limit < 0) else args.npz_limit
    cfg = TrainConfig(
        scenario_path=args.scenario,
        voxel_dir=args.voxel_dir,
        npz_limit=npz_limit,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        dim=args.dim,
        heads=args.heads,
        windows=args.windows,
        device=args.device,
        task=args.task,
        recon_weight=args.recon_weight,
        clf_weight=args.clf_weight,
        test_frac=args.test_frac,
        cv_folds=args.cv_folds,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    run_training(cfg, args)


if __name__ == "__main__":
    main()
