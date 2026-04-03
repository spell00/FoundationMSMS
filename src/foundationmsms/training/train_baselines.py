from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import zipfile

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sklearn.metrics
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..logging.logger import get_logger

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


VALID_BASELINES = (
    "majority",
    "naive_bayes",
    "logreg",
    "linear_svm",
    "random_forest",
    "xgboost",
    "cnn",
)


@dataclass
class DatasetRecords:
    dataset_id: str
    token_ids: List[np.ndarray]
    token_vals: List[np.ndarray]
    labels: np.ndarray
    groups: np.ndarray
    n_classes: int
    label_map: Dict[int, str]


@dataclass
class BaselineConfig:
    scenario: Path
    out_dir: Path
    models: Tuple[str, ...]
    folds: int
    seed: int
    hash_dim: int
    svd_dim: int
    device: str
    datasets: Optional[Tuple[str, ...]]
    cnn_epochs: int
    cnn_batch_size: int
    cnn_lr: float
    cnn_embed_dim: int
    cnn_channels: int
    cnn_max_seq_len: int
    xgb_estimators: int
    rf_estimators: int
    optuna_trials: int = 0
    optuna_timeout_sec: int = 0
    optuna_storage: Optional[str] = None
    cache_dir: Optional[Path] = None
    label_parsing_config: Optional[Path] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_label(label: object) -> Optional[str]:
    text = str(label).strip()
    if text in {"", "None", "nan", "unknown"}:
        return None
    return text


def _encode_token_doc(idx: object, hash_dim: int) -> np.ndarray:
    idx_arr = np.asarray(idx)
    if idx_arr.ndim == 2 and idx_arr.shape[1] == 2:
        if idx_arr.size == 0:
            return np.empty((0,), dtype=np.int64)
        stride = int(idx_arr[:, 1].max()) + 1
        token_ids = idx_arr[:, 0].astype(np.int64) * stride + idx_arr[:, 1].astype(np.int64)
    elif idx_arr.ndim == 1:
        token_ids = idx_arr.astype(np.int64)
    else:
        raise ValueError(f"Unexpected token shape: {idx_arr.shape}")
    return np.mod(token_ids, hash_dim).astype(np.int64, copy=False)


def _load_npz_member_with_progress(npz_path: Path, member_name: str, logger, chunk_size: int = 8 * 1024 * 1024) -> np.ndarray:
    with zipfile.ZipFile(npz_path, "r") as archive:
        names = set(archive.namelist())
        if member_name not in names:
            raise KeyError(f"{member_name} not found in {npz_path}")
        info = archive.getinfo(member_name)
        logger.info(
            "Reading %s (compressed=%.2f GB, uncompressed=%.2f GB)",
            member_name,
            info.compress_size / (1024 ** 3),
            info.file_size / (1024 ** 3),
        )
        with archive.open(info, "r") as src, io.BytesIO() as buf:
            pbar = tqdm(
                total=info.file_size,
                desc=f"loading {member_name}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                disable=not sys.stderr.isatty(),
            )
            try:
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    buf.write(chunk)
                    pbar.update(len(chunk))
                payload = buf.getvalue()
            finally:
                pbar.close()
    logger.info("Deserializing %s ...", member_name)
    array = np.load(io.BytesIO(payload), allow_pickle=True)
    logger.info("Loaded %s shape=%s dtype=%s", member_name, getattr(array, "shape", None), getattr(array, "dtype", None))
    return array


def _load_labeled_dataset_ids(label_parsing_config: Optional[Path], logger) -> Optional[frozenset]:
    """Return the set of lowercase dataset IDs that have label rules, or None if yaml not found."""
    candidates = [label_parsing_config] if label_parsing_config else []
    candidates.append(Path("configs/label_parsing.yaml"))
    for path in candidates:
        if path is not None and path.exists():
            with path.open() as f:
                data = yaml.safe_load(f)
            ids = frozenset(str(k).lower() for k in data.keys() if isinstance(data[k], dict))
            logger.info("Loaded label_parsing config from %s: %d labeled datasets: %s", path, len(ids), sorted(ids))
            return ids
    logger.warning("No label_parsing.yaml found; will fall back to per-sample label check")
    return None


def _load_scenario_binning_params(scenario_path: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Load mz_bin, mz_parent_bin, rt_bin_sec from scenario_meta.yaml in the scenario directory.
    
    Returns:
        Tuple of (mz_bin, mz_parent_bin, rt_bin_sec) or (None, None, None) if not found.
    """
    # Scenario is at scenario_dir/scenario_name.npz, so metadata is at scenario_dir/scenario_meta.yaml
    scenario_dir = scenario_path.parent
    meta_path = scenario_dir / "scenario_meta.yaml"
    
    if not meta_path.exists():
        return None, None, None
    
    try:
        with meta_path.open() as f:
            meta = yaml.safe_load(f) or {}
        return (
            meta.get("mz_bin"),
            meta.get("mz_parent_bin"),
            meta.get("rt_bin_sec")
        )
    except Exception:
        return None, None, None


def _resolve_scenario_path(scenario_path: Path) -> Path:
    """Resolve scenario path from a simple --scenario path.

    If the provided path exists, return it unchanged.
    Otherwise, try to locate:
      <parent>/mzbin_*/<stem>/<stem>.npz
    and return the first deterministic match.
    """
    if scenario_path.exists():
        return scenario_path

    parent = scenario_path.parent
    stem = scenario_path.stem
    matches = sorted(parent.glob(f"mzbin_*/{stem}/{stem}.npz"))
    if matches:
        return matches[0]
    return scenario_path


def _scenario_output_dir(cfg: BaselineConfig) -> Path:
    mz_bin, mz_parent_bin, rt_bin_sec = _load_scenario_binning_params(cfg.scenario)
    scenario_name = cfg.scenario.stem
    if mz_bin is not None and mz_parent_bin is not None and rt_bin_sec is not None:
        voxel_param_subdir = f"mzbin_{mz_bin}_mzparent_{mz_parent_bin}_rtbin_{rt_bin_sec}"
        return cfg.out_dir / voxel_param_subdir / scenario_name
    return cfg.out_dir / scenario_name


def _is_optuna_tunable(model_name: str) -> bool:
    return model_name in {"naive_bayes", "logreg", "linear_svm", "random_forest", "xgboost", "cnn"}


def _suggest_model_params(model_name: str, trial, cfg: BaselineConfig) -> Dict[str, object]:
    if model_name == "naive_bayes":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "fit_prior": trial.suggest_categorical("fit_prior", [True, False]),
        }
    if model_name == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        }
    if model_name == "linear_svm":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 1000, 5000, step=500),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model_name == "cnn":
        return {
            "cnn_epochs": trial.suggest_int("cnn_epochs", 5, max(5, cfg.cnn_epochs)),
            "cnn_batch_size": trial.suggest_categorical("cnn_batch_size", [16, 32, 64]),
            "cnn_lr": trial.suggest_float("cnn_lr", 1e-4, 5e-3, log=True),
            "cnn_embed_dim": trial.suggest_categorical("cnn_embed_dim", [64, 128, 256]),
            "cnn_channels": trial.suggest_categorical("cnn_channels", [64, 128, 256]),
        }
    return {}


def _write_optuna_trials(rows: List[Dict[str, object]], scenario_dir: Path) -> None:
    if not rows:
        return
    out_path = scenario_dir / "baselines_optuna_trials.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_labeled_scenario(cfg: BaselineConfig, logger) -> Dict[str, DatasetRecords]:
    # Try loading from cache first
    if cfg.cache_dir is not None:
        cached = _try_load_cache(cfg, logger)
        if cached is not None:
            return cached

    # Determine which dataset IDs have label rules defined in label_parsing.yaml.
    labeled_ds_ids = _load_labeled_dataset_ids(cfg.label_parsing_config, logger)

    with np.load(cfg.scenario, allow_pickle=True) as scenario:
        if "labels" not in scenario or "dataset_ids" not in scenario:
            raise ValueError("Scenario must contain labels and dataset_ids for baseline training")

        # Load cheap scalar arrays first to build a keep-mask before touching tokens.
        dataset_ids_raw = scenario["dataset_ids"]
        labels_raw = scenario["labels"]
        sample_names_raw = scenario["sample_names"] if "sample_names" in scenario else None

    # Build keep-mask: include only rows whose dataset has label rules (vectorized).
    # If yaml wasn't found, fall back to per-sample label string check.
    dataset_ids_str = np.array([str(d).lower() for d in dataset_ids_raw])
    if labeled_ds_ids is not None:
        keep_mask = np.isin(dataset_ids_str, list(labeled_ds_ids))
    else:
        keep_mask = np.array([_normalize_label(l) is not None for l in labels_raw])

    # Apply optional --datasets filter on top.
    if cfg.datasets:
        allowed = frozenset(d.lower() for d in cfg.datasets)
        keep_mask &= np.isin(dataset_ids_str, list(allowed))

    logger.info("Scenario has %d docs total, %d in labeled datasets", len(labels_raw), keep_mask.sum())
    logger.info("Loading token arrays from scenario (this may take several minutes for large files)...")

    # For large .npz files, surface byte-level progress while reading each member.
    tokens_idx = _load_npz_member_with_progress(cfg.scenario, "tokens_idx.npy", logger)
    tokens_val = _load_npz_member_with_progress(cfg.scenario, "tokens_val.npy", logger)

    # Boolean-index to keep only labeled rows, then immediately free the full arrays
    # to avoid holding 58 GB of token data and a 47 k-row copy in RAM simultaneously.
    kept_idx   = tokens_idx[keep_mask]
    kept_val   = tokens_val[keep_mask]
    del tokens_idx, tokens_val

    kept_dsids  = dataset_ids_raw[keep_mask]
    kept_labels = labels_raw[keep_mask]
    kept_names  = sample_names_raw[keep_mask] if sample_names_raw is not None else None

    grouped: Dict[str, Dict[str, list]] = {}
    n_kept = int(keep_mask.sum())
    pbar = tqdm(zip(kept_idx, kept_val, kept_dsids, kept_labels),
                total=n_kept, desc="building doc records", disable=not sys.stderr.isatty())
    for i, (idx, val, dataset_id, label) in enumerate(pbar):
        label_text = _normalize_label(label)
        if label_text is None:
            continue  # skip any samples that still lack a valid label
        idx_arr = np.asarray(idx)
        val_arr = np.asarray(val)
        if idx_arr.size == 0:
            continue
        dsid = str(dataset_id)
        sample_name = str(kept_names[i]) if kept_names is not None else f"sample_{i}"
        grouped.setdefault(dsid, {"idx": [], "val": [], "label": [], "sample": []})
        grouped[dsid]["idx"].append(_encode_token_doc(idx_arr, cfg.hash_dim))
        grouped[dsid]["val"].append(np.asarray(val_arr, dtype=np.float32))
        grouped[dsid]["label"].append(label_text)
        grouped[dsid]["sample"].append(sample_name)

    records: Dict[str, DatasetRecords] = {}
    for dsid, payload in grouped.items():
        unique_labels = sorted(set(payload["label"]))
        if len(unique_labels) < 2:
            logger.warning("Skipping dataset %s because it has fewer than 2 classes", dsid)
            continue
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        labels_encoded = np.array([label_to_id[label] for label in payload["label"]], dtype=np.int64)
        groups_arr = np.array(payload["sample"], dtype=object)
        records[dsid] = DatasetRecords(
            dataset_id=dsid,
            token_ids=payload["idx"],
            token_vals=payload["val"],
            labels=labels_encoded,
            groups=groups_arr,
            n_classes=len(unique_labels),
            label_map={idx: label for label, idx in label_to_id.items()},
        )

    if cfg.cache_dir is not None:
        _save_cache(records, cfg, logger)

    return records


def _cache_key(cfg: BaselineConfig) -> str:
    """Stable key based on scenario path, hash_dim, and dataset filter."""
    key_str = f"{cfg.scenario.resolve()}|{cfg.hash_dim}|{sorted(cfg.datasets) if cfg.datasets else 'all'}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def _try_load_cache(cfg: BaselineConfig, logger) -> Optional[Dict[str, DatasetRecords]]:
    cache_dir = cfg.cache_dir
    key = _cache_key(cfg)
    meta_path = cache_dir / f"meta_{key}.json"
    if not meta_path.exists():
        return None
    try:
        with meta_path.open() as f:
            meta = json.load(f)
        records: Dict[str, DatasetRecords] = {}
        for dsid, info in meta.items():
            sparse_path = cache_dir / f"sparse_{key}_{dsid}.npz"
            seq_path = cache_dir / f"seq_{key}_{dsid}.npz"
            if not sparse_path.exists() or not seq_path.exists():
                return None
            seq_data = np.load(seq_path, allow_pickle=True)
            token_ids_list = list(seq_data["token_ids"])
            token_vals_list = list(seq_data["token_vals"])
            records[dsid] = DatasetRecords(
                dataset_id=dsid,
                token_ids=token_ids_list,
                token_vals=token_vals_list,
                labels=seq_data["labels"],
                groups=seq_data["groups"],
                n_classes=int(info["n_classes"]),
                label_map={int(k): v for k, v in info["label_map"].items()},
            )
        logger.info("Loaded %d datasets from feature cache at %s", len(records), cache_dir)
        return records
    except Exception as exc:
        logger.warning("Cache load failed (%s), re-building features from scenario", exc)
        return None


def _save_cache(records: Dict[str, DatasetRecords], cfg: BaselineConfig, logger) -> None:
    cache_dir = cfg.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(cfg)
    meta = {}
    for dsid, rec in records.items():
        seq_path = cache_dir / f"seq_{key}_{dsid}.npz"
        np.savez_compressed(
            seq_path,
            token_ids=np.array(rec.token_ids, dtype=object),
            token_vals=np.array(rec.token_vals, dtype=object),
            labels=rec.labels,
            groups=rec.groups,
        )
        meta[dsid] = {"n_classes": rec.n_classes, "label_map": {str(k): v for k, v in rec.label_map.items()}}
    meta_path = cache_dir / f"meta_{key}.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved feature cache to %s (key=%s)", cache_dir, key)


def _dense_stats(token_docs: List[np.ndarray], value_docs: List[np.ndarray], hash_dim: int) -> np.ndarray:
    stats = np.zeros((len(token_docs), 10), dtype=np.float32)
    for idx, (token_ids, values) in enumerate(zip(token_docs, value_docs)):
        values = np.asarray(values, dtype=np.float32)
        unique_tokens = np.unique(token_ids)
        stats[idx, 0] = float(len(token_ids))
        stats[idx, 1] = float(len(unique_tokens))
        stats[idx, 2] = float(len(unique_tokens) / max(1, len(token_ids)))
        stats[idx, 3] = float(values.sum()) if values.size else 0.0
        stats[idx, 4] = float(values.mean()) if values.size else 0.0
        stats[idx, 5] = float(values.std()) if values.size else 0.0
        stats[idx, 6] = float(values.max()) if values.size else 0.0
        stats[idx, 7] = float(values.min()) if values.size else 0.0
        stats[idx, 8] = float(np.count_nonzero(values > 0) / max(1, values.size)) if values.size else 0.0
        stats[idx, 9] = float(np.bincount(token_ids, minlength=hash_dim).max()) if token_ids.size else 0.0
    return stats


def build_sparse_features(token_docs: List[np.ndarray], value_docs: List[np.ndarray], hash_dim: int) -> sparse.csr_matrix:
    row_idx: List[int] = []
    count_col_idx: List[int] = []
    count_data: List[float] = []
    sum_col_idx: List[int] = []
    sum_data: List[float] = []

    for row, (token_ids, values) in enumerate(zip(token_docs, value_docs)):
        if token_ids.size == 0:
            continue
        unique_ids, inverse = np.unique(token_ids, return_inverse=True)
        counts = np.bincount(inverse)
        weighted = np.bincount(inverse, weights=np.asarray(values, dtype=np.float32))
        row_idx.extend([row] * len(unique_ids))
        count_col_idx.extend(unique_ids.tolist())
        count_data.extend(counts.astype(np.float32).tolist())
        sum_col_idx.extend((unique_ids + hash_dim).tolist())
        sum_data.extend(weighted.astype(np.float32).tolist())

    rows = np.array(row_idx + row_idx, dtype=np.int32)
    cols = np.array(count_col_idx + sum_col_idx, dtype=np.int32)
    data = np.array(count_data + sum_data, dtype=np.float32)
    main = sparse.csr_matrix((data, (rows, cols)), shape=(len(token_docs), hash_dim * 2), dtype=np.float32)
    dense = sparse.csr_matrix(_dense_stats(token_docs, value_docs, hash_dim))
    return sparse.hstack([main, dense], format="csr", dtype=np.float32)


def build_splitter(labels: np.ndarray, groups: np.ndarray, folds: int, seed: int):
    if folds <= 1:
        idx = np.arange(len(labels))
        class_counts = np.bincount(labels)
        min_class = int(class_counts.min()) if class_counts.size else 0
        stratify = labels if min_class >= 2 else None
        train_idx, test_idx = train_test_split(
            idx,
            test_size=0.2,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        return [(train_idx, test_idx)]

    folds = max(2, min(folds, len(labels)))
    class_counts = np.bincount(labels)
    min_class = int(class_counts.min()) if class_counts.size else 0
    unique_groups = len(np.unique(groups))
    if unique_groups >= folds and min_class >= folds:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        return splitter.split(np.zeros(len(labels)), labels, groups)
    if min_class >= folds:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return splitter.split(np.zeros(len(labels)), labels)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return splitter.split(np.zeros(len(labels)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(sklearn.metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(sklearn.metrics.balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(sklearn.metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(sklearn.metrics.matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
    }


def _dense_projection(train_sparse: sparse.csr_matrix, test_sparse: sparse.csr_matrix, svd_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    max_components = min(train_sparse.shape[0] - 1, train_sparse.shape[1] - 1, svd_dim)
    if max_components < 2:
        return train_sparse.toarray(), test_sparse.toarray()
    reducer = TruncatedSVD(n_components=max_components, random_state=0)
    return reducer.fit_transform(train_sparse), reducer.transform(test_sparse)


class SequenceDataset(Dataset):
    def __init__(self, token_docs: List[np.ndarray], value_docs: List[np.ndarray], labels: np.ndarray, max_seq_len: int):
        self.token_docs = token_docs
        self.value_docs = value_docs
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        token_ids = self.token_docs[index][: self.max_seq_len] + 1
        values = self.value_docs[index][: self.max_seq_len]
        return (
            torch.as_tensor(token_ids, dtype=torch.long),
            torch.as_tensor(values, dtype=torch.float32),
            int(self.labels[index]),
        )


def collate_sequence(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, int]]):
    token_ids, values, labels = zip(*batch)
    max_len = max(item.shape[0] for item in token_ids)
    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_values = torch.zeros(len(batch), max_len, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for row, (ids, vals) in enumerate(zip(token_ids, values)):
        length = ids.shape[0]
        padded_ids[row, :length] = ids
        padded_values[row, :length] = vals
        mask[row, :length] = True
    return padded_ids, padded_values, mask, torch.as_tensor(labels, dtype=torch.long)


class SpectralCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, channels: int, n_classes: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.value_projection = nn.Linear(1, embed_dim, bias=False)
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(channels * 2, n_classes)

    def forward(self, token_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        token_emb = self.token_embedding(token_ids)
        value_emb = self.value_projection(values.unsqueeze(-1))
        encoded = (token_emb + value_emb).transpose(1, 2)
        features = self.encoder(encoded)
        avg_pool = torch.mean(features, dim=-1)
        max_pool = torch.amax(features, dim=-1)
        pooled = self.dropout(torch.cat([avg_pool, max_pool], dim=-1))
        return self.classifier(pooled)


def evaluate_cnn(model: SpectralCNN, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for token_ids, values, _mask, labels in loader:
            logits = model(token_ids.to(device), values.to(device))
            preds = torch.argmax(logits, dim=-1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return compute_metrics(np.asarray(y_true), np.asarray(y_pred))


def train_cnn_baseline(
    records: DatasetRecords,
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: BaselineConfig,
    logger,
    model_params: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model_params = model_params or {}
    cnn_batch_size = int(model_params.get("cnn_batch_size", cfg.cnn_batch_size))
    cnn_lr = float(model_params.get("cnn_lr", cfg.cnn_lr))
    cnn_embed_dim = int(model_params.get("cnn_embed_dim", cfg.cnn_embed_dim))
    cnn_channels = int(model_params.get("cnn_channels", cfg.cnn_channels))
    cnn_epochs = int(model_params.get("cnn_epochs", cfg.cnn_epochs))

    device = torch.device(cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_dataset = SequenceDataset([records.token_ids[i] for i in train_idx], [records.token_vals[i] for i in train_idx], records.labels[train_idx], cfg.cnn_max_seq_len)
    test_dataset = SequenceDataset([records.token_ids[i] for i in test_idx], [records.token_vals[i] for i in test_idx], records.labels[test_idx], cfg.cnn_max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=cnn_batch_size, shuffle=True, collate_fn=collate_sequence)
    train_eval_loader = DataLoader(train_dataset, batch_size=cnn_batch_size, shuffle=False, collate_fn=collate_sequence)
    test_loader = DataLoader(test_dataset, batch_size=cnn_batch_size, shuffle=False, collate_fn=collate_sequence)

    model = SpectralCNN(cfg.hash_dim + 1, cnn_embed_dim, cnn_channels, records.n_classes).to(device)
    class_counts = np.bincount(records.labels[train_idx], minlength=records.n_classes)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cnn_lr)

    for epoch in range(cnn_epochs):
        model.train()
        for token_ids, values, _mask, labels in train_loader:
            logits = model(token_ids.to(device), values.to(device))
            loss = loss_fn(logits, labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        logger.info("dataset=%s model=cnn fold=%d/%d epoch=%d/%d done", records.dataset_id, fold_idx + 1, cfg.folds, epoch + 1, cnn_epochs)

    train_metrics = evaluate_cnn(model, train_eval_loader, device)
    val_metrics = evaluate_cnn(model, test_loader, device)
    return train_metrics, val_metrics


def run_classical_model(
    name: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    x_train_sparse: sparse.csr_matrix,
    x_test_sparse: sparse.csr_matrix,
    x_train_dense: Optional[np.ndarray],
    x_test_dense: Optional[np.ndarray],
    cfg: BaselineConfig,
    model_params: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model_params = model_params or {}
    if name == "majority":
        model = DummyClassifier(strategy="most_frequent")
        x_train = np.zeros((len(y_train), 1))
        x_test = np.zeros((len(y_test), 1))
        model.fit(x_train, y_train)
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
    elif name == "naive_bayes":
        model = MultinomialNB(
            alpha=float(model_params.get("alpha", 1.0)),
            fit_prior=bool(model_params.get("fit_prior", True)),
        )
        model.fit(x_train_sparse[:, : cfg.hash_dim], y_train)
        y_pred_train = model.predict(x_train_sparse[:, : cfg.hash_dim])
        y_pred_test = model.predict(x_test_sparse[:, : cfg.hash_dim])
    elif name == "logreg":
        model = make_pipeline(
            MaxAbsScaler(),
            LogisticRegression(
                C=float(model_params.get("C", 1.0)),
                max_iter=500,
                class_weight="balanced",
                solver=str(model_params.get("solver", "lbfgs")),
                random_state=cfg.seed,
            ),
        )
        model.fit(x_train_dense, y_train)
        y_pred_train = model.predict(x_train_dense)
        y_pred_test = model.predict(x_test_dense)
    elif name == "linear_svm":
        model = make_pipeline(
            MaxAbsScaler(),
            LinearSVC(
                C=float(model_params.get("C", 1.0)),
                class_weight="balanced",
                max_iter=int(model_params.get("max_iter", 2000)),
                random_state=cfg.seed,
            ),
        )
        model.fit(x_train_sparse, y_train)
        y_pred_train = model.predict(x_train_sparse)
        y_pred_test = model.predict(x_test_sparse)
    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(model_params.get("n_estimators", cfg.rf_estimators)),
            random_state=cfg.seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_depth=int(model_params["max_depth"]) if model_params.get("max_depth") is not None else None,
            min_samples_split=int(model_params.get("min_samples_split", 2)),
            min_samples_leaf=int(model_params.get("min_samples_leaf", 1)),
            max_features=model_params.get("max_features", "sqrt"),
        )
        model.fit(x_train_dense, y_train)
        y_pred_train = model.predict(x_train_dense)
        y_pred_test = model.predict(x_test_dense)
    elif name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed in the active environment")
        params = {
            "n_estimators": int(model_params.get("n_estimators", cfg.xgb_estimators)),
            "max_depth": int(model_params.get("max_depth", 6)),
            "learning_rate": float(model_params.get("learning_rate", 0.05)),
            "subsample": float(model_params.get("subsample", 0.9)),
            "colsample_bytree": float(model_params.get("colsample_bytree", 0.9)),
            "reg_lambda": float(model_params.get("reg_lambda", 1.0)),
            "random_state": cfg.seed,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
        }
        if len(np.unique(y_train)) > 2:
            params.update({"objective": "multi:softmax", "num_class": int(len(np.unique(y_train)))})
        else:
            params.update({"objective": "binary:logistic"})
        model = XGBClassifier(**params)
        model.fit(x_train_dense, y_train)
        y_pred_train = model.predict(x_train_dense)
        y_pred_test = model.predict(x_test_dense)
    else:
        raise ValueError(f"Unknown model: {name}")
    return compute_metrics(y_train, np.asarray(y_pred_train)), compute_metrics(y_test, np.asarray(y_pred_test))


def aggregate_rows(rows: List[Dict[str, object]]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    keyed: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["model"]))
        keyed.setdefault(key, []).append(row)
    for (dataset, model), group_rows in keyed.items():
        metrics = {}
        for split in ("train", "val"):
            for metric_name in ("accuracy", "balanced_accuracy", "macro_f1", "mcc"):
                col = f"{split}_{metric_name}"
                values = [float(item[col]) for item in group_rows]
                metrics[col] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
        summary.setdefault(dataset, {})[model] = metrics
    return summary


def write_outputs(rows: List[Dict[str, object]], cfg: BaselineConfig) -> None:
    # Load binning parameters from scenario metadata
    mz_bin, mz_parent_bin, rt_bin_sec = _load_scenario_binning_params(cfg.scenario)
    
    # Build output path as:
    # <out-dir>/mzbin_*_mzparent_*_rtbin_*/<scenario-name>/...
    scenario_name = cfg.scenario.stem
    if mz_bin is not None and mz_parent_bin is not None and rt_bin_sec is not None:
        voxel_param_subdir = f"mzbin_{mz_bin}_mzparent_{mz_parent_bin}_rtbin_{rt_bin_sec}"
        scenario_dir = cfg.out_dir / voxel_param_subdir / scenario_name
    else:
        scenario_dir = cfg.out_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = scenario_dir / "baselines_fold_metrics.csv"
    legacy_csv_path = scenario_dir / "baseline_fold_metrics.csv"
    summary_path = scenario_dir / "baselines_summary.json"
    legacy_summary_path = scenario_dir / "baseline_summary.json"

    fieldnames = [
        "scenario",
        "dataset",
        "model",
        "fold",
        "mz_bin",
        "mz_parent_bin",
        "rt_bin_sec",
        "train_size",
        "val_size",
        "test_size",
        "n_classes",
        "train_accuracy",
        "train_balanced_accuracy",
        "train_macro_f1",
        "train_mcc",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "val_mcc",
        "best_params",
    ]
    
    # Add binning params to each row
    for row in rows:
        row["mz_bin"] = mz_bin
        row["mz_parent_bin"] = mz_parent_bin
        row["rt_bin_sec"] = rt_bin_sec
    
    def _write_csv(path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(csv_path)
    _write_csv(legacy_csv_path)

    summary = aggregate_rows(rows)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    with legacy_summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)


def parse_models(value: str) -> Tuple[str, ...]:
    models = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(models) - set(VALID_BASELINES))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown baselines: {', '.join(invalid)}")
    return models


def parse_datasets(value: str) -> Optional[Tuple[str, ...]]:
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    return items or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train classical and CNN baselines on labeled LCMS scenarios")
    parser.add_argument("--scenario", type=Path, required=True, help="Path to labeled scenario NPZ")
    parser.add_argument("--out-dir", type=Path, default=Path("logs/baselines"), help="Directory for baseline metrics")
    parser.add_argument("--models", type=parse_models, default=parse_models(",".join(VALID_BASELINES)), help="Comma-separated baseline list")
    parser.add_argument("--datasets", type=parse_datasets, default=None, help="Optional dataset ID filter")
    parser.add_argument("--folds", type=int, default=1, help="Number of CV folds (1 uses a single train/validation split)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hash-dim", type=int, default=16384, help="Hashed token feature space size")
    parser.add_argument("--svd-dim", type=int, default=128, help="Dense projection width for tree baselines")
    parser.add_argument("--device", type=str, default="auto", help="cnn device: auto, cpu, or cuda")
    parser.add_argument("--cnn-epochs", type=int, default=15, help="CNN epochs per fold")
    parser.add_argument("--cnn-batch-size", type=int, default=32, help="CNN batch size")
    parser.add_argument("--cnn-lr", type=float, default=1e-3, help="CNN learning rate")
    parser.add_argument("--cnn-embed-dim", type=int, default=128, help="CNN embedding dimension")
    parser.add_argument("--cnn-channels", type=int, default=128, help="CNN hidden channels")
    parser.add_argument("--cnn-max-seq-len", type=int, default=2048, help="Max sequence length for CNN inputs")
    parser.add_argument("--xgb-estimators", type=int, default=200, help="Number of XGBoost trees")
    parser.add_argument("--rf-estimators", type=int, default=200, help="Number of RandomForest trees")
    parser.add_argument("--optuna-trials", type=int, default=20, help="Optuna trials per (dataset, model, fold). 0 disables tuning")
    parser.add_argument("--optuna-timeout-sec", type=int, default=0, help="Optional timeout per study in seconds (0 = no timeout)")
    parser.add_argument("--optuna-storage", type=str, default=None, help="Optional Optuna storage URL (defaults to scenario-local sqlite db)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Directory to cache parsed scenario features across runs")
    parser.add_argument("--label-parsing-config", type=Path, default=None, help="Path to label_parsing.yaml; determines which datasets are labeled (defaults to configs/label_parsing.yaml)")
    return parser


def run_training(cfg: BaselineConfig) -> None:
    logger = get_logger("baseline_train")
    set_seed(cfg.seed)
    logger.info("Loading scenario from %s", cfg.scenario)
    datasets = load_labeled_scenario(cfg, logger)
    if not datasets:
        raise ValueError("No labeled datasets were found in the requested scenario")

    scenario_dir = _scenario_output_dir(cfg)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    optuna = None
    if cfg.optuna_trials > 0:
        try:
            import optuna as _optuna
            optuna = _optuna
        except Exception as exc:
            raise RuntimeError("Optuna is required when --optuna-trials > 0") from exc

    rows: List[Dict[str, object]] = []
    optuna_rows: List[Dict[str, object]] = []
    for dataset_id, records in datasets.items():
        logger.info(
            "dataset=%s docs=%d classes=%d unique_samples=%d models=%s",
            dataset_id,
            len(records.labels),
            records.n_classes,
            len(np.unique(records.groups)),
            cfg.models,
        )
        sparse_features = build_sparse_features(records.token_ids, records.token_vals, cfg.hash_dim)
        split_iter = list(build_splitter(records.labels, records.groups, cfg.folds, cfg.seed))
        for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            y_train = records.labels[train_idx]
            y_test = records.labels[test_idx]
            x_train_sparse = sparse_features[train_idx]
            x_test_sparse = sparse_features[test_idx]
            # SVD projection only needed for tree-based and dense models
            needs_dense = any(m in cfg.models for m in ("logreg", "random_forest", "xgboost"))
            if needs_dense:
                x_train_dense, x_test_dense = _dense_projection(x_train_sparse, x_test_sparse, cfg.svd_dim)
            else:
                x_train_dense = x_test_dense = None

            for model_name in cfg.models:
                logger.info("dataset=%s model=%s fold=%d/%d starting", dataset_id, model_name, fold_idx + 1, len(split_iter))
                best_params: Dict[str, object] = {}

                if optuna is not None and _is_optuna_tunable(model_name):
                    storage = cfg.optuna_storage or f"sqlite:///{scenario_dir / 'optuna_baselines.db'}"
                    study_name = f"{cfg.scenario.stem}_{dataset_id}_{model_name}_fold{fold_idx}"
                    study = optuna.create_study(
                        study_name=study_name,
                        storage=storage,
                        direction="maximize",
                        load_if_exists=True,
                    )

                    def objective(trial):
                        params = _suggest_model_params(model_name, trial, cfg)
                        if model_name == "cnn":
                            _train_metrics, _val_metrics = train_cnn_baseline(
                                records,
                                fold_idx,
                                train_idx,
                                test_idx,
                                cfg,
                                logger,
                                model_params=params,
                            )
                        else:
                            _train_metrics, _val_metrics = run_classical_model(
                                model_name,
                                y_train,
                                y_test,
                                x_train_sparse,
                                x_test_sparse,
                                x_train_dense,
                                x_test_dense,
                                cfg,
                                model_params=params,
                            )
                        trial.set_user_attr("train_macro_f1", float(_train_metrics["macro_f1"]))
                        trial.set_user_attr("val_macro_f1", float(_val_metrics["macro_f1"]))
                        trial.set_user_attr("val_accuracy", float(_val_metrics["accuracy"]))
                        return float(_val_metrics["macro_f1"])

                    logger.info(
                        "dataset=%s model=%s fold=%d/%d running Optuna (%d trials)",
                        dataset_id,
                        model_name,
                        fold_idx + 1,
                        len(split_iter),
                        cfg.optuna_trials,
                    )
                    study.optimize(
                        objective,
                        n_trials=cfg.optuna_trials,
                        timeout=cfg.optuna_timeout_sec if cfg.optuna_timeout_sec > 0 else None,
                    )
                    best_params = dict(study.best_params)

                    for t in study.trials:
                        row = {
                            "scenario": str(cfg.scenario),
                            "dataset": dataset_id,
                            "model": model_name,
                            "fold": int(fold_idx),
                            "trial_number": int(t.number),
                            "state": str(t.state),
                            "objective": float(t.value) if t.value is not None else None,
                            "val_macro_f1": t.user_attrs.get("val_macro_f1"),
                            "val_accuracy": t.user_attrs.get("val_accuracy"),
                            "train_macro_f1": t.user_attrs.get("train_macro_f1"),
                            "params_json": json.dumps(t.params, sort_keys=True),
                        }
                        for k, v in t.params.items():
                            row[f"param_{k}"] = v
                        optuna_rows.append(row)

                if model_name == "cnn":
                    train_metrics, val_metrics = train_cnn_baseline(
                        records,
                        fold_idx,
                        train_idx,
                        test_idx,
                        cfg,
                        logger,
                        model_params=best_params,
                    )
                else:
                    train_metrics, val_metrics = run_classical_model(
                        model_name,
                        y_train,
                        y_test,
                        x_train_sparse,
                        x_test_sparse,
                        x_train_dense,
                        x_test_dense,
                        cfg,
                        model_params=best_params,
                    )
                rows.append(
                    {
                        "scenario": str(cfg.scenario),
                        "dataset": dataset_id,
                        "model": model_name,
                        "fold": fold_idx,
                        "train_size": int(len(train_idx)),
                        "val_size": int(len(test_idx)),
                        "test_size": int(len(test_idx)),
                        "n_classes": int(records.n_classes),
                        "train_accuracy": float(train_metrics["accuracy"]),
                        "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
                        "train_macro_f1": float(train_metrics["macro_f1"]),
                        "train_mcc": float(train_metrics["mcc"]),
                        "val_accuracy": float(val_metrics["accuracy"]),
                        "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                        "val_macro_f1": float(val_metrics["macro_f1"]),
                        "val_mcc": float(val_metrics["mcc"]),
                        "best_params": json.dumps(best_params, sort_keys=True) if best_params else "{}",
                    }
                )
                logger.info(
                    "dataset=%s model=%s fold=%d/%d train_metrics=%s validation_size=%d validation_metrics=%s",
                    dataset_id,
                    model_name,
                    fold_idx + 1,
                    len(split_iter),
                    train_metrics,
                    int(len(test_idx)),
                    val_metrics,
                )

    write_outputs(rows, cfg)
    _write_optuna_trials(optuna_rows, scenario_dir)
    logger.info("Baseline metrics written to %s", cfg.out_dir)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    resolved_scenario = _resolve_scenario_path(args.scenario)
    cfg = BaselineConfig(
        scenario=resolved_scenario,
        out_dir=args.out_dir,
        models=args.models,
        folds=args.folds,
        seed=args.seed,
        hash_dim=args.hash_dim,
        svd_dim=args.svd_dim,
        device=args.device,
        datasets=args.datasets,
        cnn_epochs=args.cnn_epochs,
        cnn_batch_size=args.cnn_batch_size,
        cnn_lr=args.cnn_lr,
        cnn_embed_dim=args.cnn_embed_dim,
        cnn_channels=args.cnn_channels,
        cnn_max_seq_len=args.cnn_max_seq_len,
        xgb_estimators=args.xgb_estimators,
        rf_estimators=args.rf_estimators,
        optuna_trials=args.optuna_trials,
        optuna_timeout_sec=args.optuna_timeout_sec,
        optuna_storage=args.optuna_storage,
        cache_dir=args.cache_dir,
        label_parsing_config=args.label_parsing_config,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()