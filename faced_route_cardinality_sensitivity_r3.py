"""Replay one FACED DRSR fold for top-14 versus all-122 route sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from data_utils.HEDNLoader import HEDNLoader
from datasets.faced_feature import FACEDFeatureDataset
from models.RWHEDN import RWHEDN
from models.RWHEDNPseudoConditional import RWHEDNPseudoConditional
from trainers.HEDNTrainer import HEDNTrainer
from upstream_v2_residual import classification_metrics
from utils.utils import setup_seed


HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("FACED_DATA_ROOT", "data/FACED"))
STUDY = "faced_route_cardinality_sensitivity"
DESIGN_PATH = HERE / "faced_route_cardinality_sensitivity_r3_design.json"
PARENT_DESIGN_PATH = HERE / "faced_external_validation_design.json"
PARENT_ROOT = (
    HERE
    / "results"
    / "faced_external_validation"
    / "faced_external_same_model_production_r3_20260909"
    / "drsr"
)
SMOKE_TAG = "faced_route_cardinality_smoke_r3_20260912"
PRODUCTION_TAG = "faced_route_cardinality_production_r3_20260912"
CHECKPOINTS = tuple(range(50, 1001, 50))
SOURCE_FILES = (
    "faced_route_cardinality_sensitivity_r3.py",
    "faced_route_cardinality_sensitivity_r3_design.json",
    "FACED_ROUTE_CARDINALITY_SENSITIVITY_R3_CONTRACT.md",
    "faced_external_validation_design.json",
    "datasets/faced_feature.py",
    "data_utils/HEDNLoader.py",
    "models/HEDN.py",
    "models/RWHEDN.py",
    "models/RWHEDNJS.py",
    "models/RWHEDNPseudoConditional.py",
    "trainers/HEDNTrainer.py",
    "upstream_v2_residual.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def code_sha256() -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update(sha256_file(HERE / name).encode())
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


class TargetLabelVault:
    def __init__(self, labels: np.ndarray) -> None:
        self._labels = np.ascontiguousarray(labels, dtype=np.float32)
        self.state = "sealed"

    def open(self, probability_path: Path) -> np.ndarray:
        if self.state != "sealed" or not probability_path.is_file():
            raise RuntimeError("target labels opened before probability commit")
        self.state = "opened_for_posthoc_evaluation"
        return self._labels.copy()


def load_design() -> dict[str, Any]:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_DESIGN_PATH.read_text(encoding="utf-8"))
    if design.get("parent_design_sha256") != sha256_file(PARENT_DESIGN_PATH):
        raise RuntimeError("FACED parent design hash changed")
    frozen = parent["frozen_configuration"]
    expected = {
        "conditional_alignment_weight": 0.0,
        "js_relevance_weight": 0.0,
        "residual_coefficient": 0.25,
        "routing_cardinality": 14,
        "routing_temperature": 1.0,
        "source_only_warmup_iterations": 0,
        "training_iterations": 1000,
        "optimizer_seed": 42,
        "source_participants_per_fold": 122,
        "target_bank_history_retention": 0.1,
        "corrected_cluster_centres": True,
    }
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise RuntimeError("FACED frozen design differs from executable constants")
    if tuple(frozen["checkpoint_iterations"]) != CHECKPOINTS:
        raise RuntimeError("FACED checkpoint lattice mismatch")
    if parent.get("faced_performance_seen_when_frozen") is not False:
        raise RuntimeError("FACED design was not frozen before outcome access")
    if design.get("original_route_cardinality") != 14:
        raise RuntimeError("FACED original route cardinality changed")
    if design.get("sensitivity_route_cardinality") != 122:
        raise RuntimeError("FACED all-source route cardinality changed")
    if design.get("adversarial_schedule_iterations") != 1000:
        raise RuntimeError("FACED adversarial schedule horizon changed")
    if design.get("smoke_executed_iterations") != 50:
        raise RuntimeError("FACED smoke stopping point changed")
    return {"sensitivity": design, "parent": parent}


def build_args(schedule_max_iter: int = 1000) -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cuda:0"), batch_size=32, num_workers=0,
        balance_source_classes=False, balance_source_seed=123,
        use_locked_cluster_params=False, feature_dim=160, num_classes=9,
        num_sources=122, num_src_clusters=15, num_tgt_clusters=15,
        transfer_loss_type="dann", max_iter=schedule_max_iter,
        src_momentum=0.5, tgt_momentum=0.1, sra_temp=1.0,
        rel_momentum=0.9, js_relevance_weight=0.0, js_eps=1e-6,
        lr=1e-3, weight_decay=1e-5, transfer_loss_weight=1.0,
        constraint_loss_weight=0.01, early_stop=0, log_interval=50,
    )


def build_trainer(model_name: str, args: SimpleNamespace) -> HEDNTrainer:
    common = {
        "input_dim": args.feature_dim,
        "num_classes": args.num_classes,
        "transfer_loss_type": args.transfer_loss_type,
        "max_iter": args.max_iter,
        "num_src_clusters": args.num_src_clusters,
        "num_tgt_clusters": args.num_tgt_clusters,
        "num_sources": args.num_sources,
        "src_momentum": args.src_momentum,
        "tgt_momentum": args.tgt_momentum,
    }
    if model_name == "drsr":
        flags = RWHEDN.stage_flags("l1")
        flags["fix_proto"] = True
        model = RWHEDNPseudoConditional(
            **common,
            **flags,
            sra_temp=1.0,
            rel_momentum=0.9,
            js_relevance_weight=0.0,
            js_eps=1e-6,
            conditional_alignment_weight=0.0,
            conditional_eps=1e-6,
        )
    else:
        raise ValueError(model_name)
    model = model.to(args.device)
    optimizer = torch.optim.RMSprop(
        model.get_step1_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    return HEDNTrainer(
        model,
        optimizer,
        lr_scheduler=None,
        max_iter=args.max_iter,
        transfer_loss_weight=args.transfer_loss_weight,
        constraint_loss_weight=args.constraint_loss_weight,
        early_stop=0,
        log_interval=50,
        device=args.device,
    )


@torch.no_grad()
def predict_batched(model: Any, data: np.ndarray, device: torch.device) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(data), 2048):
        tensor = torch.from_numpy(data[start : start + 2048]).to(device)
        outputs.append(model.predict_proba(tensor, mode="target").numpy())
    probabilities = np.concatenate(outputs).astype(np.float32)
    probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
    return probabilities


@torch.no_grad()
def classifier_batched(model: Any, data: np.ndarray, device: torch.device) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(data), 2048):
        tensor = torch.from_numpy(data[start : start + 2048]).to(device)
        feature = model.feature_extractor(tensor)
        outputs.append(torch.softmax(model.hard_classifier(feature), dim=1).cpu().numpy())
    probabilities = np.concatenate(outputs).astype(np.float32)
    probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
    return probabilities


@torch.no_grad()
def routed_snapshot(
    model: RWHEDNPseudoConditional,
    data: np.ndarray,
    device: torch.device,
    source_subjects: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    relevance = model.src_reliability.detach().cpu().numpy().astype(np.float64)
    relevance /= relevance.sum()
    order = np.lexsort((source_subjects, -relevance))
    keep = order[:14]
    selected_weight = relevance[keep] / relevance[keep].sum()
    parts: list[np.ndarray] = []
    for start in range(0, len(data), 1024):
        tensor = torch.from_numpy(data[start : start + 1024]).to(device)
        parts.append(model.source_specific_predict_proba(tensor).numpy())
    all_probability = np.concatenate(parts, axis=1).astype(np.float32)
    selected_probability = all_probability[keep]
    top14_mixture = np.einsum(
        "s,snc->nc", selected_weight, selected_probability, optimize=True
    )
    top14_mixture /= np.clip(top14_mixture.sum(axis=1, keepdims=True), 1e-12, None)
    all_weight = relevance / relevance.sum()
    all_mixture = np.einsum(
        "s,snc->nc", all_weight, all_probability, optimize=True
    )
    all_mixture /= np.clip(all_mixture.sum(axis=1, keepdims=True), 1e-12, None)
    return (
        top14_mixture.astype(np.float32),
        selected_probability,
        source_subjects[keep].astype(np.int16),
        selected_weight.astype(np.float32),
        all_mixture.astype(np.float32),
        all_probability,
        all_weight.astype(np.float32),
    )


def train_model(
    model_name: str,
    trainer: HEDNTrainer,
    source_loader: Any,
    target_loader: Any,
    target_data: np.ndarray,
    source_subjects: np.ndarray,
    checkpoints: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], np.ndarray, float, float]:
    trainer.pre_training_processing(source_loader)
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    probabilities: list[np.ndarray] = []
    classifier_probabilities: list[np.ndarray] = []
    selected_parts: list[np.ndarray] = []
    selected_subjects: list[np.ndarray] = []
    selected_weights: list[np.ndarray] = []
    all_route_final: np.ndarray | None = None
    all_parts_final: np.ndarray | None = None
    all_weights_final: np.ndarray | None = None
    log: list[list[float]] = []
    checkpoint_set = set(checkpoints)
    torch.cuda.reset_peak_memory_stats(trainer.device)
    started = time.perf_counter()
    for iteration in range(1, trainer.max_iter + 1):
        trainer.model.train()
        try:
            src_data, src_label, src_cluster = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            src_data, src_label, src_cluster = next(source_iter)
        try:
            tgt_data, _, tgt_cluster = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            tgt_data, _, tgt_cluster = next(target_iter)
        src_data = src_data.to(trainer.device)
        src_label = src_label.to(trainer.device)
        tgt_data = tgt_data.to(trainer.device)
        values = trainer.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
        loss = values[0] + trainer.transfer_loss_weight * values[1] + trainer.constraint_loss_weight * values[2]
        trainer.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        trainer.optimizer.step()
        values = trainer.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
        trainer.fe_opt.zero_grad()
        (values[3] + values[4]).backward()
        trainer.fe_opt.step()
        losses = [float(value.detach().item()) for value in values[:5]]
        if not np.isfinite(losses).all():
            raise RuntimeError(f"non-finite loss at iteration {iteration}")
        log.append([iteration, *losses, int(values[5]), int(values[6])])
        if iteration in checkpoint_set:
            classifier_probabilities.append(
                classifier_batched(trainer.model, target_data, trainer.device)
            )
            route, parts, subjects, weights, all_route, all_parts, all_weights = routed_snapshot(
                trainer.model, target_data, trainer.device, source_subjects
            )
            probabilities.append(route)
            selected_parts.append(parts)
            selected_subjects.append(subjects)
            selected_weights.append(weights)
            if iteration == checkpoints[-1]:
                all_route_final = all_route
                all_parts_final = all_parts
                all_weights_final = all_weights
    torch.cuda.synchronize(trainer.device)
    elapsed = time.perf_counter() - started
    if all_route_final is None or all_parts_final is None or all_weights_final is None:
        raise RuntimeError("FACED final all-source snapshot was not captured")
    arrays = {
        "checkpoint_iteration": np.asarray(checkpoints, dtype=np.int16),
        "probability_trajectory": np.stack(probabilities).astype(np.float32),
        "classifier_probability_trajectory": np.stack(classifier_probabilities).astype(np.float32),
        "selected_source_probability_trajectory": np.stack(selected_parts),
        "selected_source_subject": np.stack(selected_subjects),
        "selected_source_weight": np.stack(selected_weights),
        "all_source_route_probability_final": all_route_final.astype(np.float32),
        "all_source_probability_final": all_parts_final.astype(np.float32),
        "all_source_subject": source_subjects.astype(np.int16),
        "all_source_weight_final": all_weights_final.astype(np.float32),
    }
    peak_mib = float(torch.cuda.max_memory_allocated(trainer.device) / 1024**2)
    return arrays, np.asarray(log, dtype=np.float64), elapsed, peak_mib


def evaluate(
    probability_path: Path,
    vault: TargetLabelVault,
    model_name: str,
    target_subject: int,
    phase: str,
) -> dict[str, Any]:
    with np.load(probability_path, allow_pickle=False) as artifact:
        checkpoints = artifact["checkpoint_iteration"].astype(int)
        top14_probabilities = artifact["probability_trajectory"].astype(np.float64)
        top14_routes = artifact["route_probability_trajectory"].astype(np.float64)
        classifiers = artifact["classifier_probability_trajectory"].astype(np.float64)
        all122_probability = artifact["all_source_probability_final_mixture"].astype(np.float64)
    parent_dir = PARENT_ROOT / f"target_subject_{target_subject:03d}"
    parent_probability_path = parent_dir / "probabilities.npz"
    parent_manifest_path = parent_dir / "fold_manifest.json"
    if not parent_probability_path.is_file() or not parent_manifest_path.is_file():
        raise RuntimeError("preserved FACED parent fold is incomplete")
    with np.load(parent_probability_path, allow_pickle=False) as parent:
        parent_checkpoints = parent["checkpoint_iteration"].astype(int)
        indices = [int(np.where(parent_checkpoints == checkpoint)[0][0]) for checkpoint in checkpoints]
        parent_probabilities = parent["probability_trajectory"][indices].astype(np.float64)
        parent_routes = parent["route_probability_trajectory"][indices].astype(np.float64)
        parent_classifiers = parent["classifier_probability_trajectory"][indices].astype(np.float64)
    reconstruction = {
        "pipeline_max_abs": float(np.max(np.abs(top14_probabilities - parent_probabilities))),
        "route_max_abs": float(np.max(np.abs(top14_routes - parent_routes))),
        "classifier_max_abs": float(np.max(np.abs(classifiers - parent_classifiers))),
    }
    if max(reconstruction.values()) > 5e-7:
        raise RuntimeError(f"FACED parent probability reconstruction failed: {reconstruction}")

    labels = vault.open(probability_path).argmax(axis=1).astype(np.int16)
    top14_rows = [classification_metrics(labels, matrix) for matrix in top14_probabilities]
    parent_rows = [classification_metrics(labels, matrix) for matrix in parent_probabilities]
    metric_keys = ("accuracy", "uar", "macro_f1", "ece_equal_mass_10")
    metric_max_abs = max(
        abs(float(left[key]) - float(right[key]))
        for left, right in zip(top14_rows, parent_rows, strict=True)
        for key in metric_keys
    )
    if metric_max_abs > 1e-10:
        raise RuntimeError(f"FACED parent metric reconstruction failed: {metric_max_abs}")
    all122_metrics = classification_metrics(labels, all122_probability)
    return {
        "study": STUDY,
        "phase": phase,
        "model": model_name,
        "target_subject": target_subject,
        "endpoint_checkpoint": int(checkpoints[-1]),
        "top14_metrics": top14_rows[-1],
        "all122_metrics": all122_metrics,
        "top14_checkpoint_metrics": [
            {"checkpoint": int(checkpoint), **metrics}
            for checkpoint, metrics in zip(checkpoints, top14_rows, strict=True)
        ],
        "parent_reconstruction": {
            **reconstruction,
            "metric_max_abs": float(metric_max_abs),
            "parent_probabilities_sha256": sha256_file(parent_probability_path),
            "parent_manifest_sha256": sha256_file(parent_manifest_path),
        },
        "target_labels_used_for_training_or_fixed_final_selection": False,
        "target_labels_opened_after_probability_commit": True,
        "target_label_sha256": sha256_array(labels),
    }


def run_fold(model_name: str, phase: str, target_subject: int) -> Path:
    designs = load_design()
    design = designs["sensitivity"]
    parent_design = designs["parent"]
    executed_iterations = 50 if phase == "smoke" else 1000
    checkpoints = (50,) if phase == "smoke" else CHECKPOINTS
    tag = SMOKE_TAG if phase == "smoke" else PRODUCTION_TAG
    fold_dir = (
        HERE
        / "results"
        / STUDY
        / tag
        / model_name
        / f"target_subject_{target_subject:03d}"
    )
    if fold_dir.exists():
        raise RuntimeError(f"fresh fold output already exists: {fold_dir}")
    fold_dir.mkdir(parents=True, exist_ok=False)

    setup_seed(42)
    dataset = FACEDFeatureDataset(
        DATA_ROOT, apply_lds=True, scale_per_subject=True
    ).get_dataset()
    data = np.asarray(dataset["data"], dtype=np.float32)
    labels = np.asarray(dataset["labels"], dtype=np.float32)
    groups = np.asarray(dataset["groups"], dtype=np.int16)
    target_mask = groups[:, 0] == target_subject
    source_mask = ~target_mask
    if int(target_mask.sum()) != 840:
        raise RuntimeError("FACED target lattice is not 840 windows")
    target_data = np.ascontiguousarray(data[target_mask])
    target_groups = np.ascontiguousarray(groups[target_mask])
    vault = TargetLabelVault(labels[target_mask])
    source_subjects = np.unique(groups[source_mask, 0]).astype(np.int16)
    target_zeros = np.zeros((len(target_data), 9), dtype=np.float32)

    # The adversarial schedule horizon is part of the frozen scientific model.
    # Smoke shortens execution, not that schedule.
    args = build_args(schedule_max_iter=1000)
    loader_builder = HEDNLoader(
        args,
        {
            "data": np.ascontiguousarray(data[source_mask]),
            "labels": np.ascontiguousarray(labels[source_mask]),
            "groups": np.ascontiguousarray(groups[source_mask]),
        },
        {"data": target_data, "labels": target_zeros, "groups": target_groups},
    )
    source_loader, target_loader = loader_builder()
    if len(source_subjects) != 122 or np.any(target_loader.dataset.d2.numpy()):
        raise RuntimeError("FACED label-isolated LOSO contract failed")
    trainer = build_trainer(model_name, args)
    schedule = trainer.model.hard_advcriterion.loss_func.lambda_scheduler
    if int(schedule.max_iter) != 1000 or int(schedule.curr_iter) != 0:
        raise RuntimeError("FACED frozen adversarial schedule was not initialized")
    trainer.max_iter = executed_iterations
    arrays, training_log, training_seconds, peak_mib = train_model(
        model_name,
        trainer,
        source_loader,
        target_loader,
        target_data,
        source_subjects,
        checkpoints,
    )

    if model_name == "drsr":
        route = arrays.pop("probability_trajectory").astype(np.float64)
        classifier = arrays["classifier_probability_trajectory"].astype(np.float64)
        if classifier.shape != route.shape:
            raise RuntimeError("FACED classifier and structural checkpoint lattices differ")
        pipeline = 0.75 * classifier + 0.25 * route
        pipeline /= np.clip(pipeline.sum(axis=2, keepdims=True), 1e-12, None)
        all_route = arrays["all_source_route_probability_final"].astype(np.float64)
        all_pipeline = 0.75 * classifier[-1] + 0.25 * all_route
        all_pipeline /= np.clip(all_pipeline.sum(axis=1, keepdims=True), 1e-12, None)
        arrays["route_probability_trajectory"] = route.astype(np.float32)
        arrays["probability_trajectory"] = pipeline.astype(np.float32)
        arrays["all_source_probability_final_mixture"] = all_pipeline.astype(np.float32)

    probability_path = fold_dir / "probabilities.npz"
    atomic_npz(
        probability_path,
        **arrays,
        target_subject=np.full(len(target_data), target_subject, dtype=np.int16),
        target_trial=target_groups[:, 1].astype(np.int16),
        target_window=np.tile(np.arange(1, 31, dtype=np.int16), 28),
    )
    log_path = fold_dir / "training_log.csv"
    np.savetxt(
        log_path,
        training_log,
        delimiter=",",
        header=(
            "iteration,classification,transfer,consistency,source_cluster,"
            "target_cluster,easy_source_index,hard_source_index"
        ),
        comments="",
    )
    evaluation_path = fold_dir / "evaluation.json"
    atomic_json(
        evaluation_path,
        evaluate(probability_path, vault, model_name, target_subject, phase),
    )
    manifest = {
        "study": STUDY,
        "tag": tag,
        "phase": phase,
        "model": model_name,
        "target_subject": target_subject,
        "source_subject_count": 122,
        "source_subjects": source_subjects.astype(int).tolist(),
        "seed": 42,
        "max_iter": executed_iterations,
        "adversarial_schedule_max_iter": 1000,
        "batch_size": 32,
        "feature_dim": 160,
        "num_classes": 9,
        "dbscan": {
            "eps": float(loader_builder.best_cluster_params["eps"]),
            "min_samples": int(loader_builder.best_cluster_params["min_samples"]),
        },
        "training_seconds": float(training_seconds),
        "peak_cuda_mib": peak_mib,
        "sensitivity_design_sha256": sha256_file(DESIGN_PATH),
        "parent_design_sha256": sha256_file(PARENT_DESIGN_PATH),
        "code_sha256": code_sha256(),
        "target_labels_used_for_training_or_selection": False,
        "artifacts": {
            "probabilities.npz": sha256_file(probability_path),
            "training_log.csv": sha256_file(log_path),
            "evaluation.json": sha256_file(evaluation_path),
        },
        "frozen_configuration": parent_design["frozen_configuration"],
        "route_policies": {
            "original": "top 14 by descending reliability with participant-index tie-break",
            "sensitivity": "all 122 sources with normalized final reliability",
        },
    }
    manifest_path = fold_dir / "fold_manifest.json"
    atomic_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("drsr",), required=True)
    parser.add_argument("--phase", choices=("smoke", "production"), required=True)
    parser.add_argument("--subject", type=int, required=True)
    args = parser.parse_args()
    if args.subject < 1 or args.subject > 123:
        raise ValueError("FACED target subject must be in 1..123")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(run_fold(args.model, args.phase, args.subject))


if __name__ == "__main__":
    main()
