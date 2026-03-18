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

from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import sklearn.metrics

from ..logging.logger import get_logger
from ..logging.experiment import ExperimentConfig, init_experiment, log_metric, log_hparams
from ..models import MSWConfig, MSWTransformer
import torch.nn.functional as F

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# ---------------------
# Dataset and collation
# ---------------------


class DocDataset(Dataset):
    """Document dataset over tokens_idx/tokens_val arrays.

    Supports 1D token ids or 2D (frag, rt) pairs by hashing pairs into a single
    vocabulary index. Optionally carries labels for classification.
    """


    def __init__(self, tokens_idx: Iterable, tokens_val: Iterable, labels: Iterable = None, sample_to_dataset: Optional[Iterable] = None, pair_stride: Optional[int] = None, max_seq_len: Optional[int] = None, chunk_overlap: float = 0.1):
        """
        If max_seq_len is set, documents longer than this will be split into chunks of max_seq_len.
        Each chunk inherits the same label as the original document.
        """
        idx_clean: List[np.ndarray] = []
        val_clean: List[np.ndarray] = []
        samples_clean: List[np.ndarray] = []
        label_clean: List[str] = []
        doc_origins: List[int] = []  # Track which original doc each chunk comes from
        has_labels = labels is not None
        max_seq_len = int(max_seq_len) if max_seq_len is not None else None
        doc_counter = 0
        overlap = float(chunk_overlap)
        if has_labels:
            for idx, val, lab, sample in zip(tokens_idx, tokens_val, labels, sample_to_dataset):
                idx_arr = np.asarray(idx)
                val_arr = np.asarray(val)
                sample_arr = np.asarray(sample)
                if idx_arr.size == 0:
                    doc_counter += 1
                    continue
                # Chunking logic with overlap
                if max_seq_len is not None and idx_arr.shape[0] > max_seq_len:
                    stride = int(max_seq_len * (1 - overlap))
                    if stride < 1:
                        stride = 1
                    for start in range(0, idx_arr.shape[0], stride):
                        end = min(start + max_seq_len, idx_arr.shape[0])
                        idx_clean.append(idx_arr[start:end])
                        val_clean.append(val_arr[start:end])
                        samples_clean.append(sample_arr)  
                        label_clean.append(str(lab))
                        doc_origins.append(doc_counter)
                        if end == idx_arr.shape[0]:
                            break
                else:
                    idx_clean.append(idx_arr)
                    val_clean.append(val_arr)
                    samples_clean.append(sample_arr)
                    label_clean.append(str(lab))
                    doc_origins.append(doc_counter)
                doc_counter += 1
        else:
            for idx, val, sample in zip(tokens_idx, tokens_val, sample_to_dataset):
                idx_arr = np.asarray(idx)
                val_arr = np.asarray(val)
                sample_arr = np.asarray(sample)
                if idx_arr.size == 0:
                    doc_counter += 1
                    continue
                if max_seq_len is not None and idx_arr.shape[0] > max_seq_len:
                    stride = int(max_seq_len * (1 - overlap))
                    if stride < 1:
                        stride = 1
                    for start in range(0, idx_arr.shape[0], stride):
                        end = min(start + max_seq_len, idx_arr.shape[0])
                        idx_clean.append(idx_arr[start:end])
                        val_clean.append(val_arr[start:end])
                        samples_clean.append(sample_arr[start:end])
                        doc_origins.append(doc_counter)
                        if end == idx_arr.shape[0]:
                            break
                else:
                    idx_clean.append(idx_arr)
                    val_clean.append(val_arr)
                    samples_clean.append(sample_arr)
                    doc_origins.append(doc_counter)
                doc_counter += 1

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
        self.samples_list = [np.asarray(s) for s in samples_clean]
        self.doc_origins = np.array(doc_origins, dtype=np.int64)  # For aggregation

        if has_labels:
            # Build per-dataset label mappings
            # First, determine dataset for each sample
            if sample_to_dataset is not None and len(sample_to_dataset) == len(label_clean):
                sample_datasets = [str(ds) for ds in sample_to_dataset]
            else:
                sample_datasets = ["default"] * len(label_clean)

            # Build mapping: dsid -> set of labels
            dataset_to_labels = {}
            for i, (dsid, lab) in enumerate(zip(sample_datasets, label_clean)):
                dataset_to_labels.setdefault(dsid, set()).add(lab)

            # Build per-dataset label_to_id and id_to_label
            self.label_to_id = {dsid: {lab: i for i, lab in enumerate(sorted(labels))} for dsid, labels in dataset_to_labels.items()}
            self.id_to_label = {dsid: {i: lab for lab, i in label_map.items()} for dsid, label_map in self.label_to_id.items()}

            # Encode labels per sample using per-dataset mapping
            self.labels = np.array([
                self.label_to_id[dsid][lab]
                for dsid, lab in zip(sample_datasets, label_clean)
            ], dtype=np.int64)
            self.sample_datasets = sample_datasets  # Save for __getitem__ if needed
        else:
            self.label_to_id = None
            self.id_to_label = None
            self.labels = None
            self.sample_datasets = None

    def __len__(self) -> int:
        return len(self.idx_list)

    def __getitem__(self, i: int):
        idx = self.idx_list[i]
        val = self.val_list[i]
        sample = self.samples_list[i]
        origin = self.doc_origins[i]
        # sample is a dataset name (string or np.str_), never convert to tensor
        if isinstance(sample, np.ndarray):
            if sample.ndim == 0:
                dataset_name = str(sample.item())
            elif sample.ndim == 1 and sample.size > 0:
                dataset_name = str(sample[0])
            else:
                dataset_name = "default"
        else:
            dataset_name = str(sample)
        # Robustly handle both 1D and 2D idx at runtime
        if idx.ndim == 2 and idx.shape[1] == 2:
            frag = torch.as_tensor(idx[:, 0], dtype=torch.long)
            rt = torch.as_tensor(idx[:, 1], dtype=torch.long)
            idx_tensor = frag * self.pair_stride + rt
        elif idx.ndim == 1:
            idx_tensor = torch.as_tensor(idx, dtype=torch.long)
        else:
            raise ValueError(f"Unexpected idx shape: {idx.shape}")
        val_tensor = torch.as_tensor(val, dtype=torch.float32)
        if self.labels is None:
            return idx_tensor, val_tensor, dataset_name, origin
        label = int(self.labels[i])
        # Return dataset_name so downstream can use per-dataset label indices
        return idx_tensor, val_tensor, dataset_name, label, origin


def collate_recon(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
    # Accepts (idx, val) or (idx, val, origin)
    idxs, vals = zip(*[(b[0], b[1]) for b in batch])
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
    # Accept (idx, val, label, origin, dataset) or (idx, val, label, origin) or (idx, val, label)
    if len(batch[0]) == 5:
        idxs, vals, datasets, labels, _ = zip(*batch)
    elif len(batch[0]) == 4:
        idxs, vals, labels, origins = zip(*batch)
        datasets = None
    else:
        idxs, vals, labels = zip(*batch)
        origins = None
        datasets = None
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
    if datasets is not None:
        return padded_idx, padded_val, mask, labels_tensor, cls_id, list(datasets)
    else:
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
    n_steps_per_epoch: Optional[int]
    n_steps_per_warmup_epoch: Optional[int]


class TrainingRunner:
    def __init__(self, cfg: TrainConfig, args):
        self.cfg = cfg
        self.args = args
        self.logger = get_logger("msw_train")
        self.ExperimentConfig = ExperimentConfig
        self.init_experiment = init_experiment
        self.log_metric = log_metric
        self.exp_config = ExperimentConfig(
            log_dir=args.log_dir,
            run_name=args.run_name,
            project="foundationmsms",
            use_tensorboard=True,
            use_comet=True,
        )
        self.writers = self.init_experiment(self.exp_config)
        self.device = None
        self.model = None
        self.embed = None
        self.head_recon = None
        self.head_clf = None
        self.opt = None
        self.ce_loss = None
        self.scaler = None
        self.autocast_device = None
        self.checkpoint_dir = None
        self.dsid_to_clf_name = {}
        self.dataset = None
        self.tokens_idx = None
        self.tokens_val = None
        self.labels = None
        self.sample_to_dataset = None
        self.label_values = None
        self.folds = None
        self.remain_idx = None
        self.test_idx = None
        self.test_loader = None
        self.train_loader = None
        self.val_loader = None
        self.collate_fn = None
        self.rng = None
        self.parent_bins = None
        self.kind = None
        self.loaded_warmup = False
        self.warmup_scheduler = None
        self.main_scheduler = None
        self.best_states = None
        self.epochs_no_improve = 0
        self.best_score = float("inf")
        self.batch_count = 0
        self.fold_idx = 0
        self.num_folds = 1
        self.state = None
        self.val_metrics = None
        self.val_score = None
        self.test_metrics = None
        self.epoch_iter = None
        self.batch_iter = None
        self.msg = None
        self.acc = None
        self.avg_clf_loss = None
        self.avg_recon_loss = None
        self.recon_loss_sum = 0.0
        self.clf_loss_sum = None
        self.clf_correct = None
        self.clf_total = None
        self.clf_preds = None
        self.clf_targets = None
        self.loaded_warmup = False

    def build_scheduler(self, opt, scheduler_type, total_epochs):
        if scheduler_type is None or scheduler_type.lower() == 'none':
            return None
        if scheduler_type.lower() == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs)
        if scheduler_type.lower() == 'step':
            return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, total_epochs // 3), gamma=0.1)
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def init_models(self):
        model_cfg = MSWConfig(
            dim=self.cfg.dim,
            num_heads=self.cfg.heads,
            window_sizes=self.cfg.windows
        )
        model = MSWTransformer(model_cfg)
        use_cls = self.cfg.task in {"clf", "joint"}
        embed = nn.Embedding(self.dataset.vocab + (2 if use_cls else 1), model_cfg.dim, padding_idx=0)
        head_recon: Optional[nn.Linear] = None
        head_clf_dict = {}
        if self.cfg.task in {"recon", "joint"}:
            head_recon = nn.Linear(model_cfg.dim, 1)
        if self.cfg.task in {"clf", "joint"}:
            for dsid in self.dataset_ids:
                if any(d == dsid for d in self.sample_to_dataset):
                    n_classes = len(set(l for d, l in zip(self.sample_to_dataset, self.label_values) if d == dsid))
                    if n_classes > 1:
                        head_clf_dict[dsid] = nn.Linear(model_cfg.dim, n_classes)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"[info] Using {torch.cuda.device_count()} GPUs via DataParallel")
            model = nn.DataParallel(model)
            model.to(self.device)
            embed.to(self.device)
            if head_recon is not None:
                head_recon.to(self.device)
            for head in head_clf_dict.values():
                head.to(self.device)
            print("[info] DataParallel is active. Model and all heads moved to CUDA.")
        else:
            model.to(self.device)
            embed.to(self.device)
            if head_recon is not None:
                head_recon.to(self.device)
            for head in head_clf_dict.values():
                head.to(self.device)
        params = list(model.parameters()) + list(embed.parameters())
        if head_recon is not None:
            params += list(head_recon.parameters())
        for head in head_clf_dict.values():
            params += list(head.parameters())
        opt = torch.optim.Adam(params, lr=self.cfg.lr)
        ce_loss = nn.CrossEntropyLoss()
        return model, embed, head_recon, head_clf_dict, opt, ce_loss

    def evaluate(self, loader: Optional[DataLoader], model, embed, head_recon, head_clf, ce_loss):
        if loader is None:
            return None
        model.eval()
        embed.eval()
        if head_recon is not None:
            head_recon.eval()
        if head_clf is not None:
            if isinstance(head_clf, dict):
                for head in head_clf.values():
                    head.eval()
            else:
                head_clf.eval()
        recon_loss_sum = 0.0
        clf_loss_sum = {dsid: 0.0 for dsid in head_clf.keys()}
        clf_correct = {dsid: 0 for dsid in head_clf.keys()}
        clf_total = {dsid: 0 for dsid in head_clf.keys()}
        clf_preds = {dsid: [] for dsid in head_clf.keys()}
        clf_targets = {dsid: [] for dsid in head_clf.keys()}
        acc_dict = {dsid: 0.0 for dsid in clf_correct}
        mcc_dict = {dsid: 0.0 for dsid in clf_correct}
        batch_count = 0
        with torch.no_grad():
            eval_iter = tqdm(loader, desc="Evaluating", leave=False, dynamic_ncols=True)
            for batch in eval_iter:
                if self.cfg.task in {"clf", "joint"}:
                    idx, val, mask, labels_tensor, cls_id, batch_datasets = batch
                    labels_tensor = labels_tensor.to(self.device)
                else:
                    idx, val, mask = batch
                    labels_tensor = None
                    batch_datasets = None

                idx = idx.to(self.device)
                val = val.to(self.device)
                mask = mask.to(self.device)

                tok = embed(idx)
                tok = tok.contiguous()
                out = model(tok)

                if self.cfg.task in {"recon", "joint"} and head_recon is not None:
                    pred_recon = head_recon(out).squeeze(-1)
                    recon_mask = mask
                    if self.cfg.task == "joint":
                        recon_mask = recon_mask.clone()
                        recon_mask[:, 0] = False
                    masked_pred = pred_recon[recon_mask]
                    masked_val = val[recon_mask]
                    recon_loss_sum += torch.mean((masked_pred - masked_val) ** 2).item()

                if self.cfg.task in {"clf", "joint"} and head_clf is not None and ce_loss is not None and labels_tensor is not None and batch_datasets is not None:
                    for i in range(len(idx)):
                        dsid = batch_datasets[i]
                        if dsid not in head_clf:
                            continue
                        pred_clf_i = head_clf[dsid](out[i, 0, :].unsqueeze(0))
                        label_i = labels_tensor[i].unsqueeze(0)
                        n_classes = head_clf[dsid].weight.shape[0]
                        if not (label_i >= 0 and label_i < n_classes):
                            continue
                        clf_loss_i = ce_loss(pred_clf_i, label_i)
                        if getattr(self.args, "verbose", False):
                            print(f"[DEBUG] sample {i} dsid: {dsid}")
                            print(f"[DEBUG] pred_clf_i: {pred_clf_i.detach().cpu().numpy()}")
                            print(f"[DEBUG] label_i: {label_i.item()}")
                            print(f"[DEBUG] clf_loss_i: {clf_loss_i.item()}")
                        clf_loss_sum[dsid] += clf_loss_i.item()
                        with torch.no_grad():
                            clf_correct[dsid] += (pred_clf_i.argmax(dim=-1) == label_i).sum().item()
                            clf_total[dsid] += 1

                batch_count += 1
                if batch_count % 10 == 0:
                    acc_dict = {dsid: (clf_correct[dsid] / max(1, clf_total[dsid])) for dsid in clf_correct}
                    mcc_dict = {dsid: (sklearn.metrics.matthews_corrcoef(clf_targets[dsid], clf_preds[dsid]) if len(set(clf_targets[dsid])) > 1 else 0.0) for dsid in clf_targets}
                    eval_iter.set_postfix({
                        "batches": batch_count,
                        "recon_loss": f"{recon_loss_sum / max(1, batch_count):.4f}",
                        "clf_loss": {dsid: f"{clf_loss_sum[dsid] / max(1, batch_count):.4f}" for dsid in clf_loss_sum},
                        "acc": acc_dict,
                        "mcc": mcc_dict,
                    })
        # ...existing code for returning metrics...

        metrics = {
                    "batches": batch_count,
                    "recon_loss": recon_loss_sum / max(1, batch_count),
                    "clf_loss": {dsid: (clf_loss_sum[dsid] / max(1, batch_count)) for dsid in clf_loss_sum},
                    "acc": acc_dict,
                    "mcc": mcc_dict,
                }

        return metrics

    def score_for_early_stop(self, metrics: dict) -> Optional[float]:
        if metrics is None:
            return None
        if self.cfg.task == "recon":
            return metrics.get("recon_loss")
        if self.cfg.task == "clf":
            clf = metrics.get("clf_loss")
            if clf is None:
                return None
            if isinstance(clf, dict):
                return float(np.mean([float(v) for v in clf.values()])) if clf else None
            return float(clf)
        recon = metrics.get("recon_loss")
        clf = metrics.get("clf_loss")
        if recon is None or clf is None:
            return None
        if isinstance(clf, dict):
            # Convert all values to float before averaging
            clf_value = float(np.mean([float(v) for v in clf.values()]))
        else:
            clf_value = float(clf)
        return (self.cfg.recon_weight * float(recon)) + (self.cfg.clf_weight * clf_value)

    def aggregate_chunks(self, logits, origins, strategy):
        device = logits.device
        origins = origins.cpu().numpy()
        n_classes = logits.shape[1]
        doc_ids = np.unique(origins)
        agg_logits = torch.zeros((len(doc_ids), n_classes), device=device)
        for i, doc_id in enumerate(doc_ids):
            chunk_logits = logits[origins == doc_id]
            if strategy == "vote":
                agg_logits[i] = chunk_logits.max(dim=0).values
            elif strategy == "mean":
                agg_logits[i] = chunk_logits.mean(dim=0)
            elif strategy == "logsumexp":
                agg_logits[i] = torch.logsumexp(chunk_logits, dim=0)
            elif strategy == "majority":
                preds = chunk_logits.argmax(dim=-1)
                counts = torch.bincount(preds, minlength=n_classes)
                agg_logits[i] = F.one_hot(counts.argmax(), num_classes=n_classes).float()
            else:
                agg_logits[i] = chunk_logits.max(dim=0).values
        return agg_logits, doc_ids

    # ...existing code for all methods and logic in run_training, refactored as methods...


def log_sample_flow(logger, idx: torch.Tensor, mask: torch.Tensor, tok: torch.Tensor, out: torch.Tensor, pred: torch.Tensor) -> None:
    logger.info(
        "Batch shapes idx=%s mask=%s embed=%s msw=%s head=%s",
        tuple(idx.shape),
        tuple(mask.shape),
        tuple(tok.shape),
        tuple(out.shape),
        tuple(pred.shape) if pred is not None else "None",
    )


def run_training(cfg: TrainConfig, args) -> None:
    # Aggregation functions for chunked predictions
    logger = get_logger("msw_train")
    from ..logging.experiment import ExperimentConfig, init_experiment, log_metric, log_hparams
    exp_config = ExperimentConfig(
        log_dir=args.log_dir,
        run_name=args.run_name,
        project="foundationmsms",
        use_tensorboard=True,
        use_comet=True,
    )
    writers = init_experiment(exp_config)
    # Load clf_name mapping from label_parsing.yaml for dataset IDs
    import yaml
    try:
        with open("configs/label_parsing.yaml", "r") as f:
            label_cfg = yaml.safe_load(f)
        dsid_to_clf_name = {str(dsid): v.get("clf_name", str(dsid)) for dsid, v in label_cfg.items()}
    except Exception:
        dsid_to_clf_name = {}

    # Load preprocessing params for TensorBoard HParams (parallel-coordinates view)
    preprocessing_params: dict = {}
    _preproc_cfg_path = Path(getattr(args, "preprocessing_config", "configs/preprocessing.yaml"))
    try:
        _preproc_data = yaml.safe_load(_preproc_cfg_path.read_text())
        preprocessing_params = {
            "mz_bin": float(_preproc_data.get("mz_bin", 1.0)),
            "mz_parent_bin": float(_preproc_data.get("mz_parent_bin", 1.0)),
            "rt_bin_sec": float(_preproc_data.get("rt_bin_sec", 1.0)),
            "window_sec": int(_preproc_data.get("window_sec", 30)),
            "stride_sec": int(_preproc_data.get("stride_sec", 15)),
        }
    except Exception:
        pass

    if cfg.voxel_dir:
        describe_voxel_dir(cfg.voxel_dir, cfg.npz_limit, logger)

    sc = np.load(cfg.scenario_path, allow_pickle=True)
    parent_bins = sc["parent_bins"]
    tokens_idx = sc["tokens_idx"]
    tokens_val = sc["tokens_val"]
    labels = sc["labels"] if "labels" in sc else None
    dataset_ids_arr = sc["dataset_ids"] if "dataset_ids" in sc else None
    kind = sc.get("kind", None)

    logger.info("Loaded scenario %s kind=%s docs=%d", cfg.scenario_path, kind, len(parent_bins))
    lengths = describe_docs(tokens_idx, logger)

    if cfg.task in {"clf", "joint"} and labels is None:
        raise ValueError("Classification task requires 'labels' in scenario NPZ")

    # Dataset/label summary
    dataset_ids = set()
    sample_to_dataset = []
    label_summary = {}  # Ensure label_summary is always defined
    if dataset_ids_arr is not None:
        sample_to_dataset = list(dataset_ids_arr)
        dataset_ids = set(sample_to_dataset)
    # if labels is not None and hasattr(labels[0], '__iter__') and not isinstance(labels[0], str) and dataset_ids_arr is None:
        # e.g. [(PXD012353, 0), (PXD028735, 1), ...] (legacy format)
    #     for lab in labels:
    #         dataset_ids.add(lab[0])
    #         sample_to_dataset.append(lab[0])
    #     label_values = [lab[1] for lab in labels]
        # Build label summary for each dataset
    #     for dsid in dataset_ids:
    #         ds_labels = [lab[1] for lab in labels if lab[0] == dsid]
    #         label_summary[dsid] = sorted(set(ds_labels))
    # else:
    #     dataset_ids = dataset_ids or {"default"}
    #     sample_to_dataset = sample_to_dataset or (["default"] * len(labels) if labels is not None else [])
    #     label_values = labels
    #     if labels is not None:
    #         label_summary["default"] = sorted(set(labels))

    # Print dataset usage summary
    label_values = labels if labels is not None else None
    logger.info("--- Dataset Usage Summary ---")
    logger.info("All datasets: %s", list(dataset_ids))
    if labels is not None:
        logger.info("Supervised datasets (with labels): %s", [ds for ds in dataset_ids if len(label_summary.get(ds, [])) > 0])
        logger.info("Unsupervised datasets (no labels): %s", [ds for ds in dataset_ids if len(label_summary.get(ds, [])) == 0])
        for ds in dataset_ids:
            logger.info("Dataset %s: %d samples, unique labels: %s", ds, sample_to_dataset.count(ds), label_summary.get(ds, []))
    else:
        logger.info("All datasets are unsupervised (no labels)")

    dataset = DocDataset(tokens_idx, tokens_val, labels=label_values, sample_to_dataset=sample_to_dataset, max_seq_len=args.max_seq_len, chunk_overlap=args.chunk_overlap)
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

    # CUDA check: if user requested cuda, ensure it's available
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested (--device cuda) but is not available. Aborting.")
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))


    # --- Model and optimizer initialization ---
    runner = TrainingRunner(cfg, args)
    runner.dataset = dataset
    runner.dataset_ids = dataset_ids
    runner.sample_to_dataset = sample_to_dataset
    runner.label_values = label_values
    runner.device = device
    model, embed, head_recon, head_clf, opt, ce_loss = runner.init_models()

    warmup_scheduler = runner.build_scheduler(opt, getattr(args, 'warmup_scheduler', None), args.warmup_epochs)
    main_scheduler = runner.build_scheduler(opt, getattr(args, 'main_scheduler', None), cfg.epochs)
    import os
    try:
        scaler = GradScaler('cuda') if device.type == "cuda" else None
        autocast_device = 'cuda'
    except ImportError:
        scaler = GradScaler() if device.type == "cuda" else None
        autocast_device = None
    # Try to load previous warmup/model if available
    loaded_warmup = False
    if os.path.exists(args.warmup_out):
        logger.info(f"Found previous warmup/model file: {args.warmup_out}, loading weights...")
        try:
            state = torch.load(args.warmup_out, map_location=device, weights_only=True)
            model.load_state_dict(state['model'])
            embed.load_state_dict(state['embed'])
            if head_recon is not None and state.get('head_recon') is not None:
                head_recon.load_state_dict(state['head_recon'])
            loaded_warmup = True
        except Exception as e:
            logger.warning(f"Failed to load warmup/model file: {e}")
    import os
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    global_step = 0
    resume_loaded = False
    resume_fold_idx = 0
    resume_epoch_idx = 0
    resume_best_score = None
    resume_epochs_no_improve = None

    if args.resume_checkpoint:
        if os.path.exists(args.resume_checkpoint):
            logger.info("Loading full resume checkpoint from %s", args.resume_checkpoint)
            state = torch.load(args.resume_checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state["model"])
            embed.load_state_dict(state["embed"])
            if head_recon is not None and state.get("head_recon") is not None:
                head_recon.load_state_dict(state["head_recon"])
            head_clf_state = state.get("head_clf")
            if head_clf_state and head_clf:
                for dsid, sd in head_clf_state.items():
                    if dsid in head_clf:
                        head_clf[dsid].load_state_dict(sd)
            if state.get("optimizer") is not None:
                opt.load_state_dict(state["optimizer"])
            if state.get("warmup_scheduler") is not None and warmup_scheduler is not None:
                warmup_scheduler.load_state_dict(state["warmup_scheduler"])
            if state.get("main_scheduler") is not None and main_scheduler is not None:
                main_scheduler.load_state_dict(state["main_scheduler"])
            global_step = int(state.get("global_step", 0))
            resume_fold_idx = int(state.get("fold_idx", 0))
            resume_epoch_idx = int(state.get("epoch", -1)) + 1
            resume_best_score = state.get("best_score")
            resume_epochs_no_improve = state.get("epochs_no_improve")
            resume_loaded = True
            logger.info(
                "Resumed training state fold=%d next_epoch=%d global_step=%d",
                resume_fold_idx + 1,
                resume_epoch_idx + 1,
                global_step,
            )
        else:
            logger.warning("Resume checkpoint not found: %s", args.resume_checkpoint)
    if args.warmup_epochs > 0:
        logger.info(f"Starting warmup phase: {args.warmup_epochs} epochs (unsupervised reconstruction only)")
        logger.info(f"[warmup] Using device: {device}")
        if device.type == "cuda":
            check_model = model.module if hasattr(model, "module") else model
            if not next(check_model.parameters()).is_cuda:
                raise RuntimeError("Model is not on CUDA during warmup!")
            if not next(embed.parameters()).is_cuda:
                raise RuntimeError("Embedding is not on CUDA during warmup!")
            if head_recon is not None and not next(head_recon.parameters()).is_cuda:
                raise RuntimeError("head_recon is not on CUDA during warmup!")
        dataset_warmup = DocDataset(tokens_idx, tokens_val, labels=label_values, max_seq_len=args.max_seq_len, chunk_overlap=args.chunk_overlap, sample_to_dataset=sample_to_dataset)
        indices_warmup = range(len(dataset_warmup))
        collate_fn_warmup = collate_recon
        train_loader_warmup = DataLoader(torch.utils.data.Subset(dataset_warmup, list(indices_warmup)), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn_warmup)
        model.train()
        embed.train()
        if head_recon is not None:
            head_recon.train()
        for epoch in range(args.warmup_epochs):
            recon_loss_sum = 0.0
            batch_count = 0
            pbar = tqdm(train_loader_warmup, desc=f"Warmup Epoch {epoch+1}/{args.warmup_epochs}", leave=False)
            step = 0
            for idx, val, mask in pbar:
                idx = idx.to(device)
                val = val.to(device)
                mask = mask.to(device)
                if device.type == "cuda":
                    if not idx.is_cuda or not val.is_cuda or not mask.is_cuda:
                        raise RuntimeError("Input tensors are not on CUDA during warmup!")
                if autocast_device:
                    with autocast(autocast_device):
                        tok = embed(idx)
                        out = model(tok)
                        pred_recon = head_recon(out).squeeze(-1)
                        masked_pred = pred_recon[mask]
                        masked_val = val[mask]
                        recon_loss = torch.mean((masked_pred - masked_val) ** 2)
                else:
                    with autocast():
                        tok = embed(idx)
                        out = model(tok)
                        pred_recon = head_recon(out).squeeze(-1)
                        masked_pred = pred_recon[mask]
                        masked_val = val[mask]
                        recon_loss = torch.mean((masked_pred - masked_val) ** 2)
                    tok = embed(idx)
                    out = model(tok)
                    pred_recon = head_recon(out).squeeze(-1)
                    masked_pred = pred_recon[mask]
                    masked_val = val[mask]
                    recon_loss = torch.mean((masked_pred - masked_val) ** 2)
                opt.zero_grad()
                if scaler:
                    scaler.scale(recon_loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    recon_loss.backward()
                    opt.step()

                step += 1
                batch_count += 1
                recon_loss_sum += recon_loss.item()  # <-- accumulate loss here
                pbar.set_postfix({"recon_loss": f"{recon_loss.item():.4f}"})
                if args.n_steps_per_warmup_epoch is not None and step >= args.n_steps_per_warmup_epoch:
                    break
            # Step the warmup scheduler if present (per epoch)
            if warmup_scheduler is not None:
                warmup_scheduler.step()
            if batch_count:
                avg_recon_loss = recon_loss_sum / batch_count
                logger.info(f"[warmup] epoch={epoch+1}/{args.warmup_epochs} recon_loss={avg_recon_loss:.4f} batches={batch_count}")
                log_metric(writers, "warmup/recon_loss", avg_recon_loss, step=global_step)
            global_step += 1
        warmup_ckpt_path = os.path.join(checkpoint_dir, f"warmup_{os.path.basename(args.warmup_out)}")
        torch.save({
            'model': model.state_dict(),
            'embed': embed.state_dict(),
            'head_recon': head_recon.state_dict() if head_recon is not None else None,
        }, warmup_ckpt_path)
        logger.info(f"Warmup phase complete. Model saved to {warmup_ckpt_path}")
        args.warmup_out = warmup_ckpt_path
    # --- Main training loop continues below ---
    num_folds = cfg.cv_folds if cfg.cv_folds > 1 else 1
    best_val_score_across_folds = float("inf")
    for fold_idx in range(num_folds):
        if resume_loaded and fold_idx < resume_fold_idx:
            logger.info("Skipping already completed fold %d due to resume checkpoint", fold_idx + 1)
            continue

        logger.info(f"[main] Using device: {device}")
        # Check model and optimizer are on CUDA if requested
        if device.type == "cuda":
            check_model = model.module if hasattr(model, "module") else model
            if not next(check_model.parameters()).is_cuda:
                raise RuntimeError("Model is not on CUDA during main training!")
            if not next(embed.parameters()).is_cuda:
                raise RuntimeError("Embedding is not on CUDA during main training!")
            if head_recon is not None and not next(head_recon.parameters()).is_cuda:
                raise RuntimeError("head_recon is not on CUDA during main training!")
        # Reload model from warmup before each fold, unless resuming mid-fold.
        if not (resume_loaded and fold_idx == resume_fold_idx):
            logger.info(f"Reloading model from {args.warmup_out} for main training phase")
            state = torch.load(args.warmup_out, map_location=device, weights_only=True)
            model.load_state_dict(state['model'])
            embed.load_state_dict(state['embed'])
            if head_recon is not None and state.get('head_recon') is not None:
                head_recon.load_state_dict(state['head_recon'])
        else:
            logger.info("Continuing from resumed in-fold state for fold %d", fold_idx + 1)

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

        model.train()
        embed.train()
        best_score = float("inf")
        best_states = None
        epochs_no_improve = 0
        epoch_start = 0
        if resume_loaded and fold_idx == resume_fold_idx:
            epoch_start = max(0, min(cfg.epochs, resume_epoch_idx))
            if resume_best_score is not None:
                best_score = float(resume_best_score)
            if resume_epochs_no_improve is not None:
                epochs_no_improve = int(resume_epochs_no_improve)
        epoch_iter = tqdm(range(epoch_start, cfg.epochs), desc=f"Fold {fold_idx+1}/{num_folds} Epochs", position=0, dynamic_ncols=True, mininterval=1.0)
        for epoch in epoch_iter:
            early_stop_triggered = False
            step = 0
            msg = f"[train] fold={fold_idx+1}/{num_folds} epoch={epoch+1}/{cfg.epochs} starting..."
            logger.info(msg)
            print(msg, flush=True)
            recon_loss_sum = 0.0
            clf_loss_sum = {dsid: 0.0 for dsid in head_clf.keys()}
            clf_correct = {dsid: 0 for dsid in head_clf.keys()}
            clf_total = {dsid: 0 for dsid in head_clf.keys()}
            # Initialize prediction/target storage for metrics
            clf_preds = {dsid: [] for dsid in head_clf.keys()}
            clf_targets = {dsid: [] for dsid in head_clf.keys()}
            batch_count = 0
            batch_iter = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Train Epoch {epoch+1}/{cfg.epochs}",
                position=1,
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0
            )
            for batch_idx, batch in batch_iter:

                if cfg.task in {"clf", "joint"}:
                    idx, val, mask, labels_tensor, cls_id, batch_datasets = batch
                    labels_tensor = labels_tensor.to(device)
                else:
                    idx, val, mask = batch
                    labels_tensor = None
                    batch_datasets = None

                idx = idx.to(device)
                val = val.to(device)
                mask = mask.to(device)

                if autocast_device:
                    with autocast(autocast_device):
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

                        if cfg.task in {"clf", "joint"} and head_clf is not None and ce_loss is not None and labels_tensor is not None and batch_datasets is not None:
                            # For each sample in batch, use correct head and dataset
                            for i in range(len(idx)):
                                dsid = batch_datasets[i]
                                if dsid not in head_clf:
                                    continue
                                pred_clf_i = head_clf[dsid](out[i, 0, :].unsqueeze(0))
                                label_i = labels_tensor[i].unsqueeze(0)
                                n_classes = head_clf[dsid].weight.shape[0]
                                if not (label_i >= 0 and label_i < n_classes):
                                    raise ValueError(f"Invalid label {label_i.item()} for CrossEntropyLoss (should be in [0, {n_classes-1}]) for dataset {dsid}")
                                clf_loss_i = ce_loss(pred_clf_i, label_i)
                                if getattr(args, "verbose", False):
                                    print(f"[DEBUG] sample {i} dsid: {dsid}")
                                    print(f"[DEBUG] pred_clf_i: {pred_clf_i.detach().cpu().numpy()}")
                                    print(f"[DEBUG] label_i: {label_i.item()}")
                                    print(f"[DEBUG] clf_loss_i: {clf_loss_i.item()}")
                                clf_loss_sum[dsid] += clf_loss_i.item()
                                with torch.no_grad():
                                    clf_correct[dsid] += (pred_clf_i.argmax(dim=-1) == label_i).sum().item()
                                    clf_total[dsid] += 1

                        if cfg.task == "joint":
                            # If either loss is missing, set to zero
                            if recon_loss is None:
                                recon_loss = torch.tensor(0.0, device=idx.device)
                            if clf_loss is None:
                                # If classification is missing, set loss to zero and optionally predict 'unknown' class
                                clf_loss = torch.tensor(0.0, device=idx.device)
                                # Optionally, set all predictions to 'unknown' (class 0)
                                # If head_clf has 'unknown' class, ensure pred_clf_i = 0 for all samples
                                # This is handled in the per-sample loop above if needed
                            loss = (cfg.recon_weight * recon_loss) + (cfg.clf_weight * clf_loss)
                        elif cfg.task == "clf":
                            if clf_loss is None:
                                raise RuntimeError("Classification loss missing")
                            loss = clf_loss
                        else:
                            if recon_loss is None:
                                raise RuntimeError("Reconstruction loss missing")
                            loss = recon_loss
                else:
                    with autocast():
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

                        if cfg.task in {"clf", "joint"} and head_clf is not None and ce_loss is not None and labels_tensor is not None and batch_datasets is not None:
                            # For each sample in batch, use correct head and dataset
                            for i in range(len(idx)):
                                dsid = batch_datasets[i]
                                if dsid not in head_clf:
                                    continue
                                pred_clf_i = head_clf[dsid](out[i, 0, :].unsqueeze(0))
                                label_i = labels_tensor[i].unsqueeze(0)
                                n_classes = head_clf[dsid].weight.shape[0]
                                if not (label_i >= 0 and label_i < n_classes):
                                    raise ValueError(f"Invalid label {label_i.item()} for CrossEntropyLoss (should be in [0, {n_classes-1}]) for dataset {dsid}")
                                clf_loss_i = ce_loss(pred_clf_i, label_i)
                                if getattr(args, "verbose", False):
                                    print(f"[DEBUG] sample {i} dsid: {dsid}")
                                    print(f"[DEBUG] pred_clf_i: {pred_clf_i.detach().cpu().numpy()}")
                                    print(f"[DEBUG] label_i: {label_i.item()}")
                                    print(f"[DEBUG] clf_loss_i: {clf_loss_i.item()}")
                                clf_loss_sum[dsid] += clf_loss_i.item()
                                with torch.no_grad():
                                    clf_correct[dsid] += (pred_clf_i.argmax(dim=-1) == label_i).sum().item()
                                    clf_total[dsid] += 1

                        if cfg.task == "joint":
                            # If either loss is missing, set to zero
                            if recon_loss is None:
                                recon_loss = torch.tensor(0.0, device=idx.device)
                            if clf_loss is None:
                                clf_loss = torch.tensor(0.0, device=idx.device)
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
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                step += 1
                batch_count += 1
                if step == 1:
                    sample_pred = pred_clf if cfg.task in {"clf", "joint"} else pred_recon
                    log_sample_flow(logger, idx, mask, tok, out, sample_pred)

                if args.n_steps_per_epoch is not None and step >= args.n_steps_per_epoch and args.n_steps_per_epoch > 0:
                    break

            # Step the main scheduler if present (per epoch)
            # Only step after all batches in the epoch
            if main_scheduler is not None:
                main_scheduler.step()

            if batch_count:
                avg_recon_loss = recon_loss_sum / batch_count
                if cfg.task == "clf":
                    acc = {dsid: clf_correct[dsid] / max(1, clf_total[dsid]) for dsid in clf_correct}
                    avg_clf_loss = {dsid: clf_loss_sum[dsid] / max(1, batch_count) for dsid in clf_loss_sum}
                    logger.info("fold=%d epoch=%d clf_loss=%s acc=%s batches=%d", fold_idx + 1, epoch,
                        {f"{dsid}:{dsid_to_clf_name.get(dsid, dsid)}": avg_clf_loss[dsid] for dsid in avg_clf_loss},
                        {f"{dsid}:{dsid_to_clf_name.get(dsid, dsid)}": acc[dsid] for dsid in acc},
                        batch_count)
                    for dsid in avg_clf_loss:
                        log_metric(writers, f"train/clf_loss", avg_clf_loss[dsid], step=global_step, head=dsid)
                        log_metric(writers, f"train/acc", acc[dsid], step=global_step, head=dsid)
                elif cfg.task == "joint":
                    acc = {dsid: clf_correct[dsid] / max(1, clf_total[dsid]) for dsid in clf_correct}
                    avg_clf_loss = {dsid: clf_loss_sum[dsid] / max(1, batch_count) for dsid in clf_loss_sum}
                    logger.info(
                        "fold=%d epoch=%d recon_loss=%.4f clf_loss=%s acc=%s batches=%d",
                        fold_idx + 1,
                        epoch,
                        avg_recon_loss,
                        {f"{dsid}:{dsid_to_clf_name.get(dsid, dsid)}": avg_clf_loss[dsid] for dsid in avg_clf_loss},
                        {f"{dsid}:{dsid_to_clf_name.get(dsid, dsid)}": acc[dsid] for dsid in acc},
                        batch_count,
                    )
                    log_metric(writers, "train/recon_loss", avg_recon_loss, step=global_step)
                    for dsid in avg_clf_loss:
                        log_metric(writers, f"train/clf_loss", avg_clf_loss[dsid], step=global_step, head=dsid)
                        log_metric(writers, f"train/acc", acc[dsid], step=global_step, head=dsid)
                else:
                    logger.info("fold=%d epoch=%d recon_loss=%.4f batches=%d", fold_idx + 1, epoch, avg_recon_loss, batch_count)
                    log_metric(writers, "train/recon_loss", avg_recon_loss, step=global_step)


            if val_loader is not None:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                val_metrics = runner.evaluate(val_loader, model, embed, head_recon, head_clf, ce_loss)
                if val_metrics:
                    val_score = runner.score_for_early_stop(val_metrics)
                    # Log validation metrics to TensorBoard/Comet
                    if cfg.task == "recon":
                        logger.info("fold=%d epoch=%d val_recon_loss=%.4f batches=%d", fold_idx + 1, epoch, val_metrics.get("recon_loss", 0.0), val_metrics["batches"])
                        log_metric(writers, "val/recon_loss", val_metrics.get("recon_loss", 0.0), step=global_step)
                    elif cfg.task == "clf":
                        val_clf_loss = val_metrics.get("clf_loss", {})
                        val_acc = val_metrics.get("acc", {})
                        logger.info(
                            "fold=%d epoch=%d val_clf_loss=%s val_acc=%s batches=%d",
                            fold_idx + 1,
                            epoch,
                            {f"{dsid}:{dsid_to_clf_name.get(dsid, dsid)}": float(v) for dsid, v in val_clf_loss.items()} if isinstance(val_clf_loss, dict) else float(val_clf_loss),
                            {f"{dsid}:{dsid_to_clf_name.get(dsid, dsid)}": float(v) for dsid, v in val_acc.items()} if isinstance(val_acc, dict) else float(val_acc),
                            val_metrics["batches"],
                        )
                        log_metric(writers, "val/clf_loss", val_metrics.get("clf_loss", 0.0), step=global_step)
                        log_metric(writers, "val/acc", val_metrics.get("acc", 0.0), step=global_step)
                    else:
                        val_clf_loss = val_metrics.get("clf_loss", {})
                        val_acc = val_metrics.get("acc", {})
                        logger.info(
                            "fold=%d epoch=%d val_recon_loss=%.4f val_clf_loss=%s val_acc=%s batches=%d",
                            fold_idx + 1,
                            epoch,
                            val_metrics.get("recon_loss", 0.0),
                            {f"{dsid}:{dsid_to_clf_name.get(dsid, dsid)}": float(v) for dsid, v in val_clf_loss.items()} if isinstance(val_clf_loss, dict) else float(val_clf_loss),
                            {f"{dsid}:{dsid_to_clf_name.get(dsid, dsid)}": float(v) for dsid, v in val_acc.items()} if isinstance(val_acc, dict) else float(val_acc),
                            val_metrics["batches"],
                        )
                        log_metric(writers, "val/recon_loss", val_metrics.get("recon_loss", 0.0), step=global_step)
                        log_metric(writers, "val/clf_loss", val_metrics.get("clf_loss", 0.0), step=global_step)
                        log_metric(writers, "val/acc", val_metrics.get("acc", 0.0), step=global_step)

                    global_step += 1

                    if val_score is not None:
                        if val_score + cfg.early_stop_min_delta < best_score:
                            best_score = val_score
                            epochs_no_improve = 0
                            best_states = (
                                copy.deepcopy(model.state_dict()),
                                copy.deepcopy(embed.state_dict()),
                                copy.deepcopy(head_recon.state_dict()) if head_recon is not None else None,
                                copy.deepcopy({dsid: head.state_dict() for dsid, head in head_clf.items()}) if head_clf is not None else None,
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
                            logger.warning(
                                "Training stopped early due to no improvement in validation metric for %d epochs (patience=%d). If you want full training, set --early-stop-patience 0.",
                                epochs_no_improve,
                                cfg.early_stop_patience,
                            )
                            early_stop_triggered = True

            model.train()
            embed.train()
            if head_recon is not None:
                head_recon.train()
            if head_clf is not None:
                if isinstance(head_clf, dict):
                    for head in head_clf.values():
                        head.train()
                else:
                    head_clf.train()

            # Always save a full-resume checkpoint each epoch for crash recovery.
            last_ckpt_path = os.path.join(checkpoint_dir, "last.pth")
            fold_last_ckpt_path = os.path.join(checkpoint_dir, f"last_fold{fold_idx+1}.pth")
            resume_state = {
                "model": model.state_dict(),
                "embed": embed.state_dict(),
                "head_recon": head_recon.state_dict() if head_recon is not None else None,
                "head_clf": {k: v.state_dict() for k, v in head_clf.items()} if head_clf else None,
                "optimizer": opt.state_dict(),
                "warmup_scheduler": warmup_scheduler.state_dict() if warmup_scheduler is not None else None,
                "main_scheduler": main_scheduler.state_dict() if main_scheduler is not None else None,
                "fold_idx": fold_idx,
                "epoch": epoch,
                "global_step": global_step,
                "best_score": best_score,
                "epochs_no_improve": epochs_no_improve,
                "cfg": vars(cfg),
            }
            torch.save(resume_state, last_ckpt_path)
            torch.save(resume_state, fold_last_ckpt_path)
            logger.info("Saved resume checkpoint: %s", fold_last_ckpt_path)

            if early_stop_triggered:
                break

        # Resume state applies only to one fold continuation.
        if resume_loaded and fold_idx == resume_fold_idx:
            resume_loaded = False

        # restore best states if we early-stopped or simply to use best val model
        if best_states is not None and val_loader is not None:
            model_state, embed_state, head_recon_state, head_clf_state, opt_state = best_states
            model.load_state_dict(model_state)
            embed.load_state_dict(embed_state)
            if head_recon is not None and head_recon_state is not None:
                head_recon.load_state_dict(head_recon_state)
            if head_clf is not None and head_clf_state is not None:
                for k, v in head_clf_state.items():
                    if k in head_clf:
                        head_clf[k].load_state_dict(v)
            opt.load_state_dict(opt_state)
            # Save best checkpoint for this fold
            best_ckpt_path = os.path.join(checkpoint_dir, f"best_fold{fold_idx+1}.pth")
            torch.save({
                'model': model.state_dict(),
                'embed': embed.state_dict(),
                'head_recon': head_recon.state_dict() if head_recon is not None else None,
                'head_clf': {k: v.state_dict() for k, v in head_clf.items()} if head_clf else None,
                'optimizer': opt.state_dict(),
            }, best_ckpt_path)
            logger.info(f"Best checkpoint for fold {fold_idx+1} saved to {best_ckpt_path}")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        test_metrics = runner.evaluate(test_loader, model, embed, head_recon, head_clf, ce_loss)
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
        if best_score < float("inf"):
            best_val_score_across_folds = min(best_val_score_across_folds, best_score)

    logger.info("Training complete")
    hparam_dict = {
        **preprocessing_params,
        "task": cfg.task,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "dim": cfg.dim,
        "heads": cfg.heads,
        "windows": ",".join(str(w) for w in cfg.windows),
        "warmup_epochs": args.warmup_epochs,
        "epochs": cfg.epochs,
        "cv_folds": cfg.cv_folds,
    }
    final_val_loss = best_val_score_across_folds if best_val_score_across_folds < float("inf") else 0.0
    log_hparams(writers, hparam_dict, {"hparam/val_loss": final_val_loss})


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
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers (default: 4)")
    parser.add_argument("--max-seq-len", type=int, default=8192, help="Max sequence length for model input (chunking for longer docs)")
    parser.add_argument("--chunk-overlap", type=float, default=0.1, help="Fractional overlap for chunking (0.1 = 10%% overlap)")
    parser.add_argument("--chunk-strategy", type=str, default="long-context", choices=["long-context", "vote", "mean", "logsumexp", "majority"], help="Chunk aggregation strategy for long docs")
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
    parser.add_argument("--device", type=str, default='cuda', help="Force device (cpu or cuda)")
    parser.add_argument("--task", type=str, choices=["recon", "clf", "joint"], default="joint", help="recon = masked MSE, clf = CLS classification, joint = both")
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Number of unsupervised warmup epochs (reconstruction only, before main training)")
    parser.add_argument("--warmup-out", type=str, default="warmup.pth", help="Path to save model after warmup phase")
    parser.add_argument("--recon-weight", type=float, default=1.0, help="Weight for reconstruction loss when joint task is used")
    parser.add_argument("--clf-weight", type=float, default=1.0, help="Weight for classification loss when joint task is used")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction of data to reserve for test set")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds (1 disables CV)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Epochs with no val improvement before stopping (0 disables)")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Minimum improvement in monitored val metric to reset patience")
    parser.add_argument("--log-dir", type=str, default="logs", help="Base directory for TensorBoard logs")
    parser.add_argument("--run-name", type=str, default=None, help="Optional TensorBoard run name; defaults to timestamp")
    parser.add_argument("--preprocessing-config", type=Path, default=Path("configs/preprocessing.yaml"),
        help="Path to preprocessing YAML; parameters are logged to TensorBoard HParams for parallel-coordinates comparison")
    parser.add_argument("--log-level", type=str, default="info", choices=["debug", "info", "warning", "error"], help="Logging level")
    parser.add_argument("--n-steps-per-epoch", type=int, default=None, help="Number of steps per epoch (for fast dev runs)")
    parser.add_argument("--n-steps-per-warmup-epoch", type=int, default=None, help="Number of steps per warmup epoch (for fast dev runs)")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to full training checkpoint for crash-safe resume")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose debug output")
    parser.add_argument("--warmup-scheduler", type=str, default=None, help="LR scheduler for warmup phase (e.g. 'cosine', 'step', 'none')")
    parser.add_argument("--main-scheduler", type=str, default=None, help="LR scheduler for main training phase (e.g. 'cosine', 'step', 'none'")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    import os
    print(f"[startup] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f"[startup] torch.cuda.device_count(): {torch.cuda.device_count()}")
    args = build_arg_parser().parse_args(argv)
    args.n_steps_per_warmup_epoch = None if (args.n_steps_per_warmup_epoch is not None and args.n_steps_per_warmup_epoch <= 0) else args.n_steps_per_warmup_epoch
    # Set up logger with user-specified log level
    logger = get_logger(log_level=args.log_level)
    # If you use logger elsewhere, ensure it is passed or imported as needed
    npz_limit = None if (args.npz_limit is not None and args.npz_limit < 0) else args.npz_limit
    # Treat max_seq_len <= 0 as unlimited
    max_seq_len = None if (args.max_seq_len is not None and args.max_seq_len <= 0) else args.max_seq_len
    # Build unique warmup/model filename based on dim and heads
    warmup_out_base = Path(args.warmup_out)
    warmup_out = warmup_out_base.parent / f"{warmup_out_base.stem}_dim{args.dim}_heads{args.heads}{warmup_out_base.suffix}"
    args.warmup_out = str(warmup_out)
    n_steps_per_epoch = args.n_steps_per_epoch
    if n_steps_per_epoch is not None and n_steps_per_epoch <= 0:
        n_steps_per_epoch = None  # None means do all steps per epoch (no limit)
    n_steps_per_warmup_epoch = args.n_steps_per_warmup_epoch
    if n_steps_per_warmup_epoch is not None and n_steps_per_warmup_epoch <= 0:
        n_steps_per_warmup_epoch = None  # None means do all steps per warmup epoch (no limit)
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
        n_steps_per_epoch=n_steps_per_epoch,
        n_steps_per_warmup_epoch=n_steps_per_warmup_epoch,
    )
    args.max_seq_len = max_seq_len
    run_training(cfg, args)


if __name__ == "__main__":
    main()
