from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from .signature_features import assign_call_ids, ensure_fields, save_call_features
from .data import build_feature_matrix, load_or_extract_signatures
from .calling import (
    apply_rule_postfilter,
    default_inference_config,
    estimate_global_coverage,
    extract_embeddings,
    generate_calls,
    get_device,
    load_model,
)
from .utils import save_vcf, set_seed

from .config import GraSVPipelinePreset, resolve_default_unified_scorer_path, select_grasv_preset
from .graph import (
    build_group_affinity,
    connectivity_clusters_from_affinity,
    group_signature_indices,
    partition_group_by_position,
)
from .scorer import apply_scorer, load_scorer, resolve_scorer_threshold


LOGGER = logging.getLogger(__name__)
_PROGRESS_LOG_DEBUG_STAGES = {"cluster_progress"}


@dataclass
class GraSVRunResult:
    preset_name: str
    graph_variant: str
    n_signatures: int
    n_clusters: int
    n_calls: int
    global_coverage: float | None
    graph_anchor_coverage: float
    output_vcf: str
    metadata_path: str


def _format_log_payload(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""
    return " " + " ".join(f"{key}={value}" for key, value in payload.items())


def _emit_progress(cli_args: argparse.Namespace, stage: str, **payload: Any) -> None:
    data = {
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    log_level = logging.DEBUG if stage in _PROGRESS_LOG_DEBUG_STAGES else logging.INFO
    LOGGER.log(log_level, "stage=%s%s", stage, _format_log_payload(payload))
    progress_json_path = str(getattr(cli_args, "progress_json_path", "") or "").strip()
    if progress_json_path:
        directory = os.path.dirname(progress_json_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(progress_json_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=True, indent=2, sort_keys=True)
    if getattr(cli_args, "verbose_progress", False):
        details = ", ".join(f"{key}={value}" for key, value in payload.items())
        suffix = f" {details}" if details else ""
        print(f"[grasv_stage] {stage}{suffix}", flush=True)


def _resolve_scorer_path(cli_args: argparse.Namespace) -> tuple[str | None, str]:
    if getattr(cli_args, "cluster_scorer_path", None):
        return str(cli_args.cluster_scorer_path), "cli"
    if getattr(cli_args, "disable_default_scorer", False):
        return None, "disabled"
    default_path = resolve_default_unified_scorer_path()
    if default_path:
        return default_path, "default_unified"
    return None, "none"


def _resolve_input_domain(cli_args: argparse.Namespace) -> str:
    explicit = str(getattr(cli_args, "domain", "") or "").strip().lower()
    if explicit in {"real", "sim"}:
        return explicit
    joined = " ".join(
        str(value or "")
        for value in (getattr(cli_args, "data_path", None), getattr(cli_args, "bam_path", None), getattr(cli_args, "output_dir", None))
    ).lower()
    return "sim" if "sim" in joined else "real"


def _build_runtime_args(cli_args: argparse.Namespace, preset: GraSVPipelinePreset) -> argparse.Namespace:
    cfg = default_inference_config()
    cfg.update(
        {
            "platform": cli_args.platform,
            "input_dim": cli_args.input_dim,
            "embed_dim": cli_args.embed_dim,
            "hidden_dims": cli_args.hidden_dims,
            "dropout": cli_args.dropout,
            "batch_size": cli_args.batch_size,
            "coverage_bin_size": cli_args.coverage_bin_size,
            "global_coverage": cli_args.global_coverage,
            "min_svlen": cli_args.min_svlen,
            "min_support": 1,
            "min_support_del": preset.postfilter.min_support_del,
            "min_support_ins": preset.postfilter.min_support_ins,
            "min_support_dup": preset.postfilter.min_support_dup,
            "min_support_inv": preset.postfilter.min_support_inv,
            "min_support_tra": max(
                preset.postfilter.min_support_dup,
                preset.postfilter.min_support_inv,
            ),
            "cluster_scorer_threshold": cli_args.cluster_scorer_threshold,
            "enable_rule_postfilter": preset.postfilter.enable_rule_postfilter,
            "min_cluster_median_mapq": preset.postfilter.min_cluster_median_mapq,
            "compactness_start_scale": preset.postfilter.compactness_start_scale,
            "compactness_end_scale": preset.postfilter.compactness_end_scale,
            "compactness_length_cv_scale": preset.postfilter.compactness_length_cv_scale,
            "compactness_cosine_relax": preset.postfilter.compactness_cosine_relax,
            "candidate_min_support": 1,
            "scorer_prefilter_mode": "balanced",
            "split_alleles": cli_args.split_alleles,
            "length_ratio_threshold": cli_args.length_ratio_threshold,
            "seed": cli_args.seed,
            "bam_path": cli_args.bam_path,
        }
    )
    return argparse.Namespace(**cfg)


def _cluster_one_group(
    signatures: Sequence[Any],
    embeddings: np.ndarray,
    graph_params: Any,
    *,
    svtype: str,
    indices: Sequence[int],
) -> List[List[int]]:
    group_clusters: List[List[int]] = []
    local_groups = partition_group_by_position(
        indices,
        signatures,
        max_position_gap=graph_params.max_position_gap,
        max_group_size=graph_params.max_group_size,
    )
    for local_indices in local_groups:
        affinity = build_group_affinity(
            local_indices,
            signatures,
            embeddings,
            svtype=svtype,
            params=graph_params,
        )
        local_clusters = connectivity_clusters_from_affinity(
            affinity,
            threshold=graph_params.similarity_threshold,
            min_cluster_size=graph_params.min_cluster_size,
        )
        for cluster in local_clusters:
            group_clusters.append([local_indices[local_idx] for local_idx in cluster])
    return group_clusters


_CLUSTER_WORKER_SIGNATURES: Sequence[Any] | None = None
_CLUSTER_WORKER_EMBEDDINGS: np.ndarray | None = None
_CLUSTER_WORKER_GRAPH_PARAMS: Any | None = None


def _init_cluster_worker(signatures: Sequence[Any], embeddings: np.ndarray, graph_params: Any) -> None:
    global _CLUSTER_WORKER_SIGNATURES
    global _CLUSTER_WORKER_EMBEDDINGS
    global _CLUSTER_WORKER_GRAPH_PARAMS
    _CLUSTER_WORKER_SIGNATURES = signatures
    _CLUSTER_WORKER_EMBEDDINGS = embeddings
    _CLUSTER_WORKER_GRAPH_PARAMS = graph_params


def _cluster_contig_worker(
    task: tuple[str, list[tuple[int, str, list[int]]]],
) -> tuple[str, int, int, dict[int, list[list[int]]]]:
    if _CLUSTER_WORKER_SIGNATURES is None or _CLUSTER_WORKER_EMBEDDINGS is None or _CLUSTER_WORKER_GRAPH_PARAMS is None:
        raise RuntimeError("Cluster worker was not initialized.")
    contig, group_tasks = task
    result_by_group: dict[int, list[list[int]]] = {}
    n_clusters = 0
    for group_index, svtype, indices in group_tasks:
        clusters = _cluster_one_group(
            _CLUSTER_WORKER_SIGNATURES,
            _CLUSTER_WORKER_EMBEDDINGS,
            _CLUSTER_WORKER_GRAPH_PARAMS,
            svtype=svtype,
            indices=indices,
        )
        result_by_group[group_index] = clusters
        n_clusters += len(clusters)
    return contig, len(group_tasks), n_clusters, result_by_group


def _cluster_signatures_parallel_by_contig(
    signatures: Sequence[Any],
    embeddings: np.ndarray,
    preset: GraSVPipelinePreset,
    group_items: list[tuple[tuple[str, str], list[int]]],
    *,
    processes: int,
    progress_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> List[List[int]]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    contig_order: list[str] = []
    tasks_by_contig: dict[str, list[tuple[int, str, list[int]]]] = {}
    for group_index, ((contig, svtype), indices) in enumerate(group_items):
        if contig not in tasks_by_contig:
            tasks_by_contig[contig] = []
            contig_order.append(contig)
        tasks_by_contig[contig].append((group_index, svtype, list(indices)))

    contig_tasks = [(contig, tasks_by_contig[contig]) for contig in contig_order]
    worker_count = max(1, min(int(processes), len(contig_tasks)))
    if progress_callback is not None:
        progress_callback(
            {
                "mode": "parallel_by_contig",
                "total_contigs": len(contig_tasks),
                "total_groups": len(group_items),
                "processes": worker_count,
            }
        )

    start_method = "fork" if "fork" in mp.get_all_start_methods() else None
    mp_context = mp.get_context(start_method) if start_method else mp.get_context()
    result_by_group: dict[int, list[list[int]]] = {}
    completed_contigs = 0
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
        initializer=_init_cluster_worker,
        initargs=(signatures, embeddings, preset.graph),
    ) as executor:
        futures = [executor.submit(_cluster_contig_worker, task) for task in contig_tasks]
        for future in as_completed(futures):
            contig, n_groups, n_clusters, contig_results = future.result()
            result_by_group.update(contig_results)
            completed_contigs += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "mode": "parallel_by_contig",
                        "completed_contigs": completed_contigs,
                        "total_contigs": len(contig_tasks),
                        "contig": contig,
                        "contig_groups": n_groups,
                        "contig_clusters": n_clusters,
                    }
                )

    clusters: List[List[int]] = []
    for group_index in range(len(group_items)):
        clusters.extend(result_by_group.get(group_index, []))
    return clusters


def _cluster_signatures_grasv(
    signatures: Sequence[Any],
    embeddings: np.ndarray,
    preset: GraSVPipelinePreset,
    *,
    include_tra: bool,
    processes: int = 1,
    progress_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> List[List[int]]:
    clusters: List[List[int]] = []
    groups = group_signature_indices(signatures, include_tra=include_tra)
    group_items = list(groups.items())
    total_groups = len(group_items)
    if int(processes) > 1 and total_groups > 1:
        return _cluster_signatures_parallel_by_contig(
            signatures,
            embeddings,
            preset,
            group_items,
            processes=int(processes),
            progress_callback=progress_callback,
        )

    for group_index, ((contig, svtype), indices) in enumerate(group_items, start=1):
        if progress_callback is not None and (
            group_index == 1 or group_index == total_groups or group_index % 50 == 0
        ):
            progress_callback(
                {
                    "group_index": group_index,
                    "total_groups": total_groups,
                    "contig": contig,
                    "svtype": svtype,
                    "group_size": len(indices),
                }
            )
        clusters.extend(
            _cluster_one_group(
                signatures,
                embeddings,
                preset.graph,
                svtype=svtype,
                indices=indices,
            )
        )

    return clusters


def _write_metadata(output_dir: str, payload: Dict[str, Any]) -> str:
    path = os.path.join(output_dir, "grasv_run_metadata.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    return path


def run_grasv_inference(cli_args: argparse.Namespace) -> GraSVRunResult:
    os.makedirs(cli_args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cli_args.output_vcf) or ".", exist_ok=True)
    set_seed(cli_args.seed)
    _emit_progress(
        cli_args,
        "start",
        platform=cli_args.platform,
        output_dir=cli_args.output_dir,
        bam_path=cli_args.bam_path,
        data_path=cli_args.data_path,
    )

    if cli_args.global_coverage is None and cli_args.bam_path:
        cli_args.global_coverage = estimate_global_coverage(cli_args.bam_path)
        if cli_args.global_coverage is not None:
            _emit_progress(cli_args, "estimated_coverage", global_coverage=round(float(cli_args.global_coverage), 4))
        else:
            LOGGER.warning("could not estimate global coverage from BAM; using coverage-aware preset fallback")

    preset = select_grasv_preset(cli_args.platform, cli_args.global_coverage)
    runtime_args = _build_runtime_args(cli_args, preset)
    run_domain = _resolve_input_domain(cli_args)
    _emit_progress(
        cli_args,
        "preset_selected",
        preset_name=preset.name,
        graph_variant=preset.graph_variant,
        graph_anchor_coverage=preset.graph_anchor_coverage,
        run_domain=run_domain,
    )

    device = get_device()
    _emit_progress(cli_args, "device_ready", device=str(device))
    encoder = load_model(cli_args, device)
    _emit_progress(cli_args, "encoder_loaded", input_dim=getattr(encoder, "input_dim", None))
    scorer_path, scorer_path_source = _resolve_scorer_path(cli_args)
    scorer_model = load_scorer(scorer_path) if scorer_path else None
    if scorer_model is None:
        LOGGER.warning("no CNN scorer checkpoint configured; calls will only use rule/postfilter stages")
    scorer_threshold_source = "preset"
    if scorer_model is not None:
        runtime_args.cluster_scorer_threshold, scorer_threshold_source = resolve_scorer_threshold(
            scorer_model,
            platform=cli_args.platform,
            override=cli_args.cluster_scorer_threshold,
        )
    elif cli_args.cluster_scorer_threshold is not None:
        runtime_args.cluster_scorer_threshold = float(cli_args.cluster_scorer_threshold)
        scorer_threshold_source = "cli_override"
    else:
        runtime_args.cluster_scorer_threshold = float(preset.postfilter.cluster_scorer_threshold)
    _emit_progress(
        cli_args,
        "models_resolved",
        scorer_path=scorer_path,
        scorer_path_source=scorer_path_source,
        scorer_threshold=runtime_args.cluster_scorer_threshold,
        scorer_threshold_source=scorer_threshold_source,
    )

    _emit_progress(cli_args, "load_signatures_start")
    signatures = load_or_extract_signatures(
        data_path=cli_args.data_path,
        bam_path=cli_args.bam_path,
        save_path=cli_args.save_signatures_path,
        chrom=cli_args.chrom,
        start=cli_args.start,
        end=cli_args.end,
        processes=cli_args.processes,
        min_sv_size=cli_args.min_sv_size,
        max_sv_size=cli_args.max_sv_size,
        min_mapq=cli_args.min_mapq,
        min_read_len=cli_args.min_read_len,
        min_siglength=cli_args.min_siglength,
        merge_del_threshold=cli_args.merge_del_threshold,
        merge_ins_threshold=cli_args.merge_ins_threshold,
        max_split_parts=cli_args.max_split_parts,
        region_size=cli_args.region_size,
    )
    if not signatures:
        raise ValueError("No signatures were loaded or extracted.")
    ensure_fields(signatures)
    _emit_progress(cli_args, "load_signatures_done", n_signatures=len(signatures))

    feature_dim = int(getattr(encoder, "input_dim", cli_args.input_dim))
    _emit_progress(
        cli_args,
        "build_features_start",
        feature_dim=feature_dim,
        coverage_cache_path=getattr(cli_args, "coverage_cache_path", None),
    )
    features = build_feature_matrix(
        signatures,
        bam_path=cli_args.bam_path,
        bin_size=cli_args.coverage_bin_size,
        platform=cli_args.platform,
        feature_dim=feature_dim,
        coverage_cache_path=getattr(cli_args, "coverage_cache_path", None),
    )
    _emit_progress(cli_args, "build_features_done", n_rows=int(features.shape[0]), n_cols=int(features.shape[1]))
    embed_started = time.time()
    _emit_progress(cli_args, "extract_embeddings_start", batch_size=cli_args.batch_size)
    embeddings = extract_embeddings(features, encoder, device, cli_args.batch_size)
    embed_seconds = time.time() - embed_started
    _emit_progress(cli_args, "extract_embeddings_done", embed_seconds=round(embed_seconds, 4))

    if cli_args.save_embeddings:
        embeddings_path = os.path.join(cli_args.output_dir, "embeddings.npy")
        np.save(embeddings_path, embeddings)
        _emit_progress(cli_args, "save_embeddings_done", path=embeddings_path)

    cluster_started = time.time()
    cluster_processes = max(1, int(getattr(cli_args, "processes", 1) or 1))
    _emit_progress(cli_args, "cluster_start", processes=cluster_processes)
    clusters = _cluster_signatures_grasv(
        signatures,
        embeddings,
        preset,
        include_tra=cli_args.include_tra,
        processes=cluster_processes,
        progress_callback=lambda payload: _emit_progress(cli_args, "cluster_progress", **payload),
    )
    cluster_seconds = time.time() - cluster_started
    _emit_progress(cli_args, "cluster_done", n_clusters=len(clusters), cluster_seconds=round(cluster_seconds, 4))

    _emit_progress(cli_args, "generate_calls_start")
    calls, filter_stats = generate_calls(clusters, signatures, embeddings, runtime_args)
    assign_call_ids(calls)
    for call in calls:
        setattr(call, "domain", run_domain)
    _emit_progress(cli_args, "generate_calls_done", n_calls=len(calls), filter_stats=dict(filter_stats))
    if cli_args.save_call_features_path:
        _emit_progress(cli_args, "save_call_features_start", path=cli_args.save_call_features_path)
        save_call_features(
            calls,
            cli_args.save_call_features_path,
            platform=cli_args.platform,
            global_coverage=cli_args.global_coverage,
            domain=run_domain,
        )
        _emit_progress(cli_args, "save_call_features_done", path=cli_args.save_call_features_path)

    if scorer_model is not None:
        n_calls_before_scorer = len(calls)
        _emit_progress(cli_args, "apply_scorer_start", n_calls=n_calls_before_scorer)
        calls, scorer_stats = apply_scorer(
            calls,
            scorer_model,
            threshold=runtime_args.cluster_scorer_threshold,
        )
        for key, value in scorer_stats.items():
            filter_stats[key] = filter_stats.get(key, 0) + int(value)
        _emit_progress(
            cli_args,
            "apply_scorer_done",
            n_calls=len(calls),
            n_removed=n_calls_before_scorer - len(calls),
            scorer_stats=dict(scorer_stats),
        )

    if runtime_args.enable_rule_postfilter:
        n_calls_before_rules = len(calls)
        _emit_progress(cli_args, "rule_postfilter_start", n_calls=n_calls_before_rules)
        calls, rule_stats = apply_rule_postfilter(calls, runtime_args)
        for key, value in rule_stats.items():
            filter_stats[key] = filter_stats.get(key, 0) + int(value)
        _emit_progress(
            cli_args,
            "rule_postfilter_done",
            n_calls=len(calls),
            n_removed=n_calls_before_rules - len(calls),
            rule_stats=dict(rule_stats),
        )

    _emit_progress(cli_args, "save_vcf_start", path=cli_args.output_vcf)
    save_vcf(calls, cli_args.output_vcf)
    _emit_progress(cli_args, "save_vcf_done", path=cli_args.output_vcf)

    metadata_path = _write_metadata(
        cli_args.output_dir,
        {
            "preset_name": preset.name,
            "graph_variant": preset.graph_variant,
            "platform": cli_args.platform,
            "domain": run_domain,
            "requested_coverage": preset.requested_coverage,
            "graph_anchor_coverage": preset.graph_anchor_coverage,
            "global_coverage": cli_args.global_coverage,
            "graph": asdict(preset.graph),
            "postfilter": asdict(preset.postfilter),
            "effective_cluster_scorer_threshold": runtime_args.cluster_scorer_threshold,
            "cluster_scorer_threshold_source": scorer_threshold_source,
            "cluster_scorer_path": scorer_path,
            "cluster_scorer_path_source": scorer_path_source,
            "n_signatures": len(signatures),
            "n_clusters": len(clusters),
            "n_calls": len(calls),
            "embedding_seconds": round(embed_seconds, 4),
            "clustering_seconds": round(cluster_seconds, 4),
            "cluster_processes": cluster_processes,
            "cluster_parallel_mode": "by_contig" if cluster_processes > 1 else "none",
            "filter_stats": filter_stats,
            "include_tra": bool(cli_args.include_tra),
        },
    )
    _emit_progress(
        cli_args,
        "done",
        metadata_path=metadata_path,
        n_signatures=len(signatures),
        n_clusters=len(clusters),
        n_calls=len(calls),
    )

    return GraSVRunResult(
        preset_name=preset.name,
        graph_variant=preset.graph_variant,
        n_signatures=len(signatures),
        n_clusters=len(clusters),
        n_calls=len(calls),
        global_coverage=cli_args.global_coverage,
        graph_anchor_coverage=preset.graph_anchor_coverage,
        output_vcf=cli_args.output_vcf,
        metadata_path=metadata_path,
    )
