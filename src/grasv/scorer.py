from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .signature_features import (
    CLUSTER_SCORER_FEATURE_NAMES,
    vectorize_call_features,
)
from .utils import get_device


class CandidateCNNBackbone(nn.Module):
    """1D CNN backbone for candidate-level tabular features."""

    def __init__(
        self,
        feature_dim: int,
        channels: tuple[int, int, int] = (32, 64, 64),
    ):
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.feature_dim = int(feature_dim)
        self.channels = tuple(int(v) for v in channels)
        self.output_dim = c3 * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected [batch, feature_dim] input, got shape={tuple(x.shape)}")
        x = x.unsqueeze(1)
        x = self.net(x)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return x.flatten(start_dim=1)


class CandidateCNNScorer(nn.Module):
    """Single-head CNN used only for candidate call filtering."""

    def __init__(
        self,
        feature_dim: int,
        channels: tuple[int, int, int] = (32, 64, 64),
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = CandidateCNNBackbone(feature_dim=feature_dim, channels=channels)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.feature_dim = int(feature_dim)
        self.channels = tuple(int(v) for v in channels)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features).squeeze(-1)


GRASV_SCORER_FEATURE_NAMES: Tuple[str, ...] = CLUSTER_SCORER_FEATURE_NAMES + (
    "domain_real",
    "domain_sim",
)


def infer_record_domain(record: Dict[str, Any]) -> str:
    domain = str(record.get("domain", "") or "").strip().lower()
    if domain in {"real", "sim"}:
        return domain
    source_tokens = " ".join(
        str(record.get(key, "") or "")
        for key in ("dataset_name", "__source_name__", "source_name", "call_id")
    ).lower()
    if "sim" in source_tokens:
        return "sim"
    return "real"


def vectorize_record_features(
    record: Dict[str, Any],
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    names = tuple(feature_names or GRASV_SCORER_FEATURE_NAMES)
    feature_dict = dict(record.get("score_features", {}) or {})
    domain = infer_record_domain(record)
    if "domain_real" in names:
        feature_dict["domain_real"] = 1.0 if domain == "real" else 0.0
    if "domain_sim" in names:
        feature_dict["domain_sim"] = 1.0 if domain == "sim" else 0.0
    return vectorize_call_features(feature_dict, feature_names=names).astype(np.float32)


def _normalize_cnn_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(key.startswith("backbone.") for key in state_dict):
        return state_dict

    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("net."):
            normalized[f"backbone.{key}"] = value
        elif key.startswith("head.1."):
            normalized[f"head.0.{key[len('head.1.'):]}"] = value
        elif key.startswith("head.4."):
            normalized[f"head.3.{key[len('head.4.'):]}"] = value
        else:
            normalized[key] = value
    return normalized


def _build_scorer_model_from_payload(model_payload: Dict[str, Any], device: torch.device) -> CandidateCNNScorer:
    model_cfg = dict(model_payload.get("model_config", {}) or {})
    model = CandidateCNNScorer(
        feature_dim=int(model_cfg.get("feature_dim", len(model_payload.get("feature_names", ())))),
        channels=tuple(model_cfg.get("channels", [32, 64, 64])),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        dropout=float(model_cfg.get("dropout", model_payload.get("dropout", 0.1))),
    )
    model.load_state_dict(_normalize_cnn_state_dict_keys(model_payload["state_dict"]))
    model.to(device)
    model.eval()
    return model


def _normalize_feature_matrix(model_payload: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    mean = np.asarray(model_payload["mean"], dtype=np.float32)
    scale = np.asarray(model_payload["scale"], dtype=np.float32)
    return ((X - mean) / np.clip(scale, 1e-6, None)).astype(np.float32)


def load_scorer(path: str) -> Dict[str, Any]:
    expanded_path = os.path.expanduser(str(path))
    payload = torch.load(expanded_path, map_location="cpu", weights_only=False)
    if str(payload.get("model_type", "")).lower() != "cnn_scorer_v1":
        raise ValueError("GraSV inference requires a cnn_scorer_v1 checkpoint.")
    return payload


def apply_scorer(
    calls: Sequence[Any],
    model_payload: Dict[str, Any],
    *,
    threshold: float | None = None,
) -> Tuple[List[Any], Dict[str, int]]:
    decision_threshold = float(model_payload.get("threshold", 0.5) if threshold is None else threshold)
    feature_names = tuple(model_payload["feature_names"])
    feature_matrix = np.asarray(
        [
            vectorize_record_features(
                {
                    "score_features": dict(getattr(call, "score_features", {}) or {}),
                    "domain": getattr(call, "domain", None),
                },
                feature_names=feature_names,
            )
            for call in calls
        ],
        dtype=np.float32,
    )
    if feature_matrix.size == 0:
        return [], {}

    device = get_device()
    model = _build_scorer_model_from_payload(model_payload, device)
    normalized = _normalize_feature_matrix(model_payload, feature_matrix)
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(normalized), 1024):
            batch = torch.from_numpy(normalized[start : start + 1024]).to(device)
            batch_probs = torch.sigmoid(model(batch)).cpu().numpy().astype(np.float32)
            probs.append(batch_probs)
    probs_arr = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)

    kept: List[Any] = []
    stats: Counter[str] = Counter()
    for call, score in zip(calls, probs_arr):
        score = float(score)
        setattr(call, "scorer_prob", score)
        setattr(call, "prob", max(float(getattr(call, "prob", 0.0) or 0.0), score))
        if score < decision_threshold:
            svtype = str(getattr(call, "svtype", "UNK")).lower()
            stats["scorer"] += 1
            stats[f"scorer_{svtype}"] += 1
            continue
        kept.append(call)
    return kept, dict(stats)


def resolve_scorer_threshold(
    model_payload: Dict[str, Any],
    *,
    platform: str | None = None,
    override: float | None = None,
) -> Tuple[float, str]:
    if override is not None:
        return float(override), "cli_override"

    platform_key = None if platform is None else str(platform).lower()
    platform_thresholds = dict(model_payload.get("platform_thresholds", {}) or {})
    if platform_key and platform_key in platform_thresholds:
        return float(platform_thresholds[platform_key]), f"platform_threshold:{platform_key}"

    if "threshold" in model_payload:
        return float(model_payload["threshold"]), "scorer_payload"

    return 0.5, "default"


__all__ = [
    "apply_scorer",
    "CandidateCNNScorer",
    "infer_record_domain",
    "load_scorer",
    "resolve_scorer_threshold",
    "vectorize_record_features",
]
