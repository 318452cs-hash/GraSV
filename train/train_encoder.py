#!/usr/bin/env python3
"""Train a GraSV-compatible VICReg encoder.

This is the public training entry point for ``models/grasv_encoder.pt``. It
trains only the self-supervised encoder used during inference. The projector
and auxiliary SVTYPE head are training-only and are not required by
``grasv infer``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grasv.calling import DEFAULT_EMBED_DIM, DEFAULT_ENCODER_HIDDEN_DIMS, DEFAULT_INPUT_DIM, GraSVEncoder  # noqa: E402
from grasv.data import build_feature_matrix, load_or_extract_signatures  # noqa: E402
from grasv.signature import guess_svtype_from_signature, normalize_svtype  # noqa: E402
from grasv.signature_features import NODE_FEAT_DIM, NODE_FEAT_NAMES, normalize_platform  # noqa: E402
from grasv.utils import get_device, set_seed  # noqa: E402


SVTYPE_TO_ID = {"DEL": 0, "INS": 1, "DUP": 2, "INV": 3, "TRA": 4, "UNK": 5}
IGNORE_SVTYPE = SVTYPE_TO_ID["UNK"]


class VICRegProjector(nn.Module):
    """Projection head used only for VICReg training."""

    def __init__(
        self,
        embed_dim: int = DEFAULT_EMBED_DIM,
        proj_dim: int = 256,
        hidden_dims: Sequence[int] = (256, 512),
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = int(embed_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.GELU())
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, int(proj_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VICRegTrainingModel(nn.Module):
    """Encoder + training-only projector and optional SVTYPE head."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: Sequence[int],
        projector_hidden_dims: Sequence[int],
        proj_dim: int,
        dropout: float,
        num_svtypes: int = 6,
    ):
        super().__init__()
        self.encoder = GraSVEncoder(
            input_dim=int(input_dim),
            embed_dim=int(embed_dim),
            hidden_dims=[int(v) for v in hidden_dims],
            dropout=float(dropout),
        )
        self.projector = VICRegProjector(
            embed_dim=int(embed_dim),
            proj_dim=int(proj_dim),
            hidden_dims=[int(v) for v in projector_hidden_dims],
        )
        self.svtype_head = nn.Linear(int(embed_dim), int(num_svtypes)) if int(num_svtypes) > 0 else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        embeddings = self.encoder(x)
        projections = self.projector(embeddings)
        logits = self.svtype_head(embeddings) if self.svtype_head is not None else None
        return embeddings, projections, logits


class AugmentedFeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Return two feature-space augmented views for VICReg training."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        noise_scale: float,
        feature_dropout: float,
    ):
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.noise_scale = float(noise_scale)
        self.feature_dropout = float(feature_dropout)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        if self.noise_scale > 0.0:
            out = out + torch.randn_like(out) * self.noise_scale
        if self.feature_dropout > 0.0:
            keep = torch.rand_like(out).ge(self.feature_dropout).float()
            out = out * keep
        return out

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.features[index]
        return self._augment(x), self._augment(x), self.labels[index]


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        if "sources" in payload or "records" in payload or "runs" in payload:
            values = payload.get("sources", payload.get("records", payload.get("runs", [])))
            return [dict(item) for item in values]
        return [dict(payload)]
    if isinstance(payload, list):
        return [dict(item) for item in payload]

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows.append(dict(json.loads(stripped)))
    return rows


def _read_manifest(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return []
    manifest_path = Path(path).expanduser()
    suffix = manifest_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return _read_json_or_jsonl(manifest_path)

    delimiter = "\t" if suffix in {".tsv", ".tab"} else ","
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def _direct_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not args.signatures_pkl and not args.bam_path:
        return []
    return [
        {
            "split": "train",
            "name": args.name or "direct_train_source",
            "data_path": args.signatures_pkl,
            "bam_path": args.bam_path,
            "feature_bam_path": args.feature_bam_path,
            "save_signatures_path": args.save_signatures_path,
            "platform": args.platform,
            "chrom": args.chrom,
            "start": args.start,
            "end": args.end,
            "max_records": args.max_records,
        }
    ]


def _split_name(row: Dict[str, Any]) -> str:
    return str(row.get("split", "train") or "train").strip().lower()


def _row_path(row: Dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value:
            return os.path.expanduser(str(value))
    return None


def _row_int(row: Dict[str, Any], name: str, default: int | None = None) -> int | None:
    value = row.get(name, default)
    if value in (None, ""):
        return default
    return int(value)


def _row_float(row: Dict[str, Any], name: str, default: float | None = None) -> float | None:
    value = row.get(name, default)
    if value in (None, ""):
        return default
    return float(value)


def _svtype_id(sig: Any) -> int:
    svtype = normalize_svtype(getattr(sig, "svtype", None)) or normalize_svtype(guess_svtype_from_signature(sig)) or "UNK"
    return SVTYPE_TO_ID.get(svtype, IGNORE_SVTYPE)


def _maybe_subsample(
    features: np.ndarray,
    labels: np.ndarray,
    max_records: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_records is None or int(max_records) <= 0 or len(features) <= int(max_records):
        return features, labels
    idx = rng.choice(len(features), size=int(max_records), replace=False)
    idx.sort()
    return features[idx], labels[idx]


def _load_split_features(
    rows: Sequence[Dict[str, Any]],
    split: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    feature_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    source_stats: List[Dict[str, Any]] = []
    selected_rows = [row for row in rows if _split_name(row) == split]
    if not selected_rows:
        raise ValueError(f"No rows found for split={split!r}.")

    extract_keys = {
        "chrom",
        "start",
        "end",
        "processes",
        "min_sv_size",
        "max_sv_size",
        "min_mapq",
        "min_read_len",
        "min_siglength",
        "merge_del_threshold",
        "merge_ins_threshold",
        "max_split_parts",
        "region_size",
    }
    for idx, row in enumerate(selected_rows):
        name = str(row.get("name") or row.get("sample_id") or f"{split}_{idx + 1}")
        platform = normalize_platform(row.get("platform") or args.platform)
        data_path = _row_path(row, "data_path", "signatures_pkl", "signatures")
        bam_path = _row_path(row, "bam_path", "bam")
        feature_bam_path = _row_path(row, "feature_bam_path") or bam_path
        save_path = _row_path(row, "save_signatures_path")
        if not data_path and not bam_path:
            raise ValueError(f"Manifest row {name!r} needs data_path/signatures_pkl or bam_path.")

        extract_kwargs: Dict[str, Any] = {}
        for key in extract_keys:
            if key in row and row[key] not in (None, ""):
                extract_kwargs[key] = row[key]
        if args.chrom and "chrom" not in extract_kwargs:
            extract_kwargs["chrom"] = args.chrom
        if args.start is not None and "start" not in extract_kwargs:
            extract_kwargs["start"] = args.start
        if args.end is not None and "end" not in extract_kwargs:
            extract_kwargs["end"] = args.end

        print(f"[load] split={split} source={name} platform={platform or 'unknown'}", flush=True)
        signatures = load_or_extract_signatures(
            data_path=data_path,
            bam_path=bam_path,
            save_path=save_path,
            **extract_kwargs,
        )
        for sig in signatures:
            if platform:
                setattr(sig, "platform", platform)
        labels = np.asarray([_svtype_id(sig) for sig in signatures], dtype=np.int64)
        features = build_feature_matrix(
            signatures,
            bam_path=feature_bam_path if args.use_bam_depth_features else None,
            bin_size=int(args.coverage_bin_size),
            platform=platform,
            feature_dim=int(args.input_dim),
        )
        max_records = _row_int(row, "max_records", args.max_records)
        features, labels = _maybe_subsample(features, labels, max_records, rng)
        if len(features) == 0:
            continue
        feature_chunks.append(features)
        label_chunks.append(labels)
        source_stats.append(
            {
                "split": split,
                "name": name,
                "platform": platform,
                "n_signatures": int(len(signatures)),
                "n_used": int(len(features)),
                "feature_shape": [int(v) for v in features.shape],
                "data_path": data_path,
                "bam_path": bam_path,
            }
        )
        print(f"[loaded] split={split} source={name} used={len(features)}", flush=True)

    if not feature_chunks:
        raise ValueError(f"No features loaded for split={split!r}.")
    return np.concatenate(feature_chunks, axis=0), np.concatenate(label_chunks, axis=0), source_stats


def _normalize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0).astype(np.float32)
    scale = X.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return ((X - mean) / scale).astype(np.float32), mean, scale


def _apply_normalization(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((X - mean) / np.clip(scale, 1e-6, None)).astype(np.float32)


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("Expected a square matrix.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    lambda_var: float,
    lambda_cov: float,
    gamma: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    inv_loss = F.mse_loss(z1, z2)

    std_z1 = torch.sqrt(z1.var(dim=0, unbiased=False) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0, unbiased=False) + 1e-4)
    var_loss = torch.mean(F.relu(float(gamma) - std_z1)) + torch.mean(F.relu(float(gamma) - std_z2))

    batch_size = z1.shape[0]
    dim = z1.shape[1]
    if batch_size > 1:
        z1_centered = z1 - z1.mean(dim=0, keepdim=True)
        z2_centered = z2 - z2.mean(dim=0, keepdim=True)
        cov_z1 = (z1_centered.T @ z1_centered) / (batch_size - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (batch_size - 1)
        cov_loss = _off_diagonal(cov_z1).pow(2).sum() / dim + _off_diagonal(cov_z2).pow(2).sum() / dim
    else:
        cov_loss = z1.new_tensor(0.0)

    total = inv_loss + float(lambda_var) * var_loss + float(lambda_cov) * cov_loss
    return total, {
        "loss": float(total.detach().cpu().item()),
        "invariance": float(inv_loss.detach().cpu().item()),
        "variance": float(var_loss.detach().cpu().item()),
        "covariance": float(cov_loss.detach().cpu().item()),
    }


def _svtype_loss(logits: torch.Tensor | None, labels: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
    if logits is None:
        return labels.new_tensor(0.0, dtype=torch.float32), {"svtype_loss": 0.0, "svtype_acc": 0.0}
    loss = F.cross_entropy(logits, labels, ignore_index=IGNORE_SVTYPE)
    valid = labels != IGNORE_SVTYPE
    if int(valid.sum().item()) == 0:
        return loss, {"svtype_loss": float(loss.detach().cpu().item()), "svtype_acc": 0.0}
    pred = logits.argmax(dim=1)
    acc = (pred[valid] == labels[valid]).float().mean()
    return loss, {"svtype_loss": float(loss.detach().cpu().item()), "svtype_acc": float(acc.detach().cpu().item())}


def _embedding_stats(model: VICRegTrainingModel, X: np.ndarray, batch_size: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(X), int(batch_size)):
            batch = torch.from_numpy(X[start : start + int(batch_size)]).to(device)
            emb = F.normalize(model.encoder(batch), p=2, dim=1)
            chunks.append(emb.cpu())
    if not chunks:
        return {"embedding_std_mean": 0.0, "embedding_abs_mean": 0.0}
    embeddings = torch.cat(chunks, dim=0)
    return {
        "embedding_std_mean": float(embeddings.std(dim=0, unbiased=False).mean().item()),
        "embedding_abs_mean": float(embeddings.abs().mean().item()),
    }


def _save_checkpoint(
    path: Path,
    model: VICRegTrainingModel,
    mean: np.ndarray,
    scale: np.ndarray,
    args: argparse.Namespace,
    history: List[Dict[str, float]],
    source_stats: List[Dict[str, Any]],
    *,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "grasv_encoder_v1",
        "epoch": int(epoch),
        "model_config": {
            "input_dim": int(args.input_dim),
            "embed_dim": int(args.embed_dim),
            "hidden_dims": [int(v) for v in args.hidden_dims],
            "dropout": float(args.dropout),
        },
        "encoder_state_dict": {key: value.detach().cpu() for key, value in model.encoder.state_dict().items()},
        "feature_names": NODE_FEAT_NAMES[: int(args.input_dim)],
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "train_history": history,
        "source_stats": source_stats,
        "training_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "lambda_var": float(args.lambda_var),
            "lambda_cov": float(args.lambda_cov),
            "gamma": float(args.gamma),
            "noise_scale": float(args.noise_scale),
            "feature_dropout": float(args.feature_dropout),
            "svtype_aux_weight": float(args.svtype_aux_weight),
            "normalize_features": bool(args.normalize_features),
        },
    }
    torch.save(payload, path)


def _write_json(path: str | None, payload: Dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray | None,
    args: argparse.Namespace,
    source_stats: List[Dict[str, Any]],
) -> tuple[VICRegTrainingModel, np.ndarray, np.ndarray, List[Dict[str, float]]]:
    if bool(args.normalize_features):
        X_train_norm, mean, scale = _normalize_features(X_train)
        X_val_norm = None if X_validation is None else _apply_normalization(X_validation, mean, scale)
    else:
        X_train_norm = np.asarray(X_train, dtype=np.float32)
        X_val_norm = None if X_validation is None else np.asarray(X_validation, dtype=np.float32)
        mean = np.zeros((X_train_norm.shape[1],), dtype=np.float32)
        scale = np.ones((X_train_norm.shape[1],), dtype=np.float32)

    device = get_device()
    model = VICRegTrainingModel(
        input_dim=int(args.input_dim),
        embed_dim=int(args.embed_dim),
        hidden_dims=[int(v) for v in args.hidden_dims],
        projector_hidden_dims=[int(v) for v in args.projector_hidden_dims],
        proj_dim=int(args.proj_dim),
        dropout=float(args.dropout),
        num_svtypes=0 if float(args.svtype_aux_weight) <= 0.0 else 6,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))

    dataset = AugmentedFeatureDataset(
        X_train_norm,
        y_train,
        noise_scale=float(args.noise_scale),
        feature_dropout=float(args.feature_dropout),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=len(dataset) >= int(args.batch_size),
    )
    history: List[Dict[str, float]] = []
    latest_path = Path(args.checkpoint_dir) / "grasv_encoder_latest.pt" if args.checkpoint_dir else None
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        accum: Counter[str] = Counter()
        n_batches = 0
        for view1, view2, labels in loader:
            view1 = view1.to(device)
            view2 = view2.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            emb1, proj1, logits1 = model(view1)
            _emb2, proj2, _logits2 = model(view2)
            loss, parts = vicreg_loss(
                proj1,
                proj2,
                lambda_var=float(args.lambda_var),
                lambda_cov=float(args.lambda_cov),
                gamma=float(args.gamma),
            )
            aux_loss, aux_parts = _svtype_loss(logits1, labels)
            loss = loss + float(args.svtype_aux_weight) * aux_loss
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            for key, value in {**parts, **aux_parts}.items():
                accum[key] += float(value)
            accum["total_loss"] += float(loss.detach().cpu().item())
            n_batches += 1
        scheduler.step()

        row = {
            "epoch": float(epoch),
            "lr": float(scheduler.get_last_lr()[0]),
            **{key: float(value / max(1, n_batches)) for key, value in accum.items()},
        }
        if X_val_norm is not None and (epoch == 1 or epoch == int(args.epochs) or epoch % int(args.eval_every) == 0):
            row.update({f"val_{k}": v for k, v in _embedding_stats(model, X_val_norm, int(args.eval_batch_size), device).items()})
        history.append(row)
        print(
            "epoch={epoch:.0f} loss={loss:.6f} inv={inv:.6f} var={var:.6f} cov={cov:.6f} aux={aux:.6f}".format(
                epoch=row["epoch"],
                loss=row.get("total_loss", 0.0),
                inv=row.get("invariance", 0.0),
                var=row.get("variance", 0.0),
                cov=row.get("covariance", 0.0),
                aux=row.get("svtype_loss", 0.0),
            ),
            flush=True,
        )
        if latest_path and (epoch % int(args.save_every) == 0 or epoch == int(args.epochs)):
            _save_checkpoint(latest_path, model, mean, scale, args, history, source_stats, epoch=epoch)

    return model, mean, scale, history


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a GraSV VICReg encoder checkpoint.")
    parser.add_argument("--manifest", default=None, help="CSV/TSV/JSON/JSONL manifest with split,data_path/platform columns.")
    parser.add_argument("--signatures-pkl", "--signatures_pkl", dest="signatures_pkl", default=None)
    parser.add_argument("--bam-path", "--bam_path", dest="bam_path", default=None)
    parser.add_argument("--feature-bam-path", "--feature_bam_path", dest="feature_bam_path", default=None)
    parser.add_argument("--save-signatures-path", "--save_signatures_path", dest="save_signatures_path", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--platform", default=None, choices=["ont", "ccs", "clr"])
    parser.add_argument("--chrom", default=None, help="Optional comma-separated contigs, e.g. 1,2,3 or chr1,chr2.")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-records", "--max_records", dest="max_records", type=int, default=None)
    parser.add_argument("--use-bam-depth-features", "--use_bam_depth_features", dest="use_bam_depth_features", action="store_true")
    parser.add_argument("--coverage-bin-size", "--coverage_bin_size", dest="coverage_bin_size", type=int, default=1000)

    parser.add_argument("--output-path", "--output_path", dest="output_path", required=True)
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", default=None)
    parser.add_argument("--summary-path", "--summary_path", dest="summary_path", default=None)

    parser.add_argument("--input-dim", "--input_dim", dest="input_dim", type=int, default=NODE_FEAT_DIM)
    parser.add_argument("--embed-dim", "--embed_dim", dest="embed_dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--hidden-dims", "--hidden_dims", dest="hidden_dims", type=int, nargs="*", default=DEFAULT_ENCODER_HIDDEN_DIMS)
    parser.add_argument("--proj-dim", "--proj_dim", dest="proj_dim", type=int, default=256)
    parser.add_argument("--projector-hidden-dims", "--projector_hidden_dims", dest="projector_hidden_dims", type=int, nargs="*", default=(256, 512))
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", "--eval_batch_size", dest="eval_batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda-var", "--lambda_var", dest="lambda_var", type=float, default=25.0)
    parser.add_argument("--lambda-cov", "--lambda_cov", dest="lambda_cov", type=float, default=25.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--noise-scale", "--noise_scale", dest="noise_scale", type=float, default=0.05)
    parser.add_argument("--feature-dropout", "--feature_dropout", dest="feature_dropout", type=float, default=0.1)
    parser.add_argument("--svtype-aux-weight", "--svtype_aux_weight", dest="svtype_aux_weight", type=float, default=0.1)
    parser.add_argument(
        "--normalize-features",
        "--normalize_features",
        dest="normalize_features",
        action="store_true",
        help="Apply z-score normalization during training. Off by default because grasv infer feeds raw node features to the encoder.",
    )
    parser.add_argument("--grad-clip", "--grad_clip", dest="grad_clip", type=float, default=5.0)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--torch-num-threads", "--torch_num_threads", dest="torch_num_threads", type=int, default=8)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=5)
    parser.add_argument("--save-every", "--save_every", dest="save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if int(args.input_dim) not in {DEFAULT_INPUT_DIM, NODE_FEAT_DIM}:
        raise SystemExit(f"Unsupported --input-dim {args.input_dim}; expected {NODE_FEAT_DIM}.")
    torch.set_num_threads(max(1, int(args.torch_num_threads)))
    set_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    rows = _read_manifest(args.manifest) + _direct_records(args)
    if not rows:
        raise SystemExit("Provide --manifest or --signatures-pkl/--bam-path.")

    X_train, y_train, train_stats = _load_split_features(rows, "train", args, rng)
    validation_rows = [row for row in rows if _split_name(row) == "validation"]
    X_val = None
    val_stats: List[Dict[str, Any]] = []
    if validation_rows:
        X_val, _y_val, val_stats = _load_split_features(rows, "validation", args, rng)

    print(
        f"[train_data] n_train={len(X_train)} dim={X_train.shape[1]} "
        f"svtype_counts={dict(Counter(int(v) for v in y_train.tolist()))}",
        flush=True,
    )
    model, mean, scale, history = _train(X_train, y_train, X_val, args, train_stats + val_stats)

    output_path = Path(args.output_path)
    _save_checkpoint(output_path, model, mean, scale, args, history, train_stats + val_stats, epoch=int(args.epochs))
    summary = {
        "output_path": str(output_path),
        "n_train": int(len(X_train)),
        "feature_dim": int(X_train.shape[1]),
        "svtype_counts": dict(Counter(int(v) for v in y_train.tolist())),
        "history": history,
        "source_stats": train_stats + val_stats,
        "device": str(get_device()),
    }
    _write_json(args.summary_path, summary)
    print(f"model={output_path}")
    if args.summary_path:
        print(f"summary={args.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
