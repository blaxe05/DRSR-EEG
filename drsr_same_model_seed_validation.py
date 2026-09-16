"""Run one locked corrected-centre, same-model DRSR fold on SEED/SEED-IV."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import faced_external_validation as shared
from data_utils._get_dataset import get_dataset
from data_utils.HEDNLoader import HEDNLoader
from seed_series_upstream_oracle import DATASET_CONTRACT, build_args, validate_dataset, within_trial_index
from upstream_v2_residual import classification_metrics
from utils.utils import setup_seed


HERE = Path(__file__).resolve().parent
STUDY = "drsr_same_model_validation"
DESIGN_PATH = HERE / "drsr_same_model_seed_design.json"
CHECKPOINTS = tuple(range(50, 1001, 50))
TAGS = {
    ("seed3", "smoke"): "drsr_same_model_seed_smoke_r1_20260910",
    ("seed4", "smoke"): "drsr_same_model_seediv_smoke_r1_20260910",
    ("seed3", "production"): "drsr_same_model_seed_production_r1_20260910",
    ("seed4", "production"): "drsr_same_model_seediv_production_r1_20260910",
}
SOURCE_FILES = (
    "drsr_same_model_seed_validation.py",
    "drsr_same_model_seed_design.json",
    "DRSR_SAME_MODEL_SEED_VALIDATION_CONTRACT.md",
    "faced_external_validation.py",
    "data_utils/HEDNLoader.py",
    "data_utils/_get_dataset.py",
    "models/HEDN.py",
    "models/RWHEDN.py",
    "models/RWHEDNJS.py",
    "models/RWHEDNPseudoConditional.py",
    "trainers/HEDNTrainer.py",
    "seed_series_upstream_oracle.py",
    "upstream_v2_residual.py",
    "config.py",
    "hedn.yaml",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_sha256() -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(sha256_file(HERE / name).encode("ascii"))
    return digest.hexdigest()


def load_design() -> dict[str, Any]:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    frozen = design["frozen_configuration"]
    required = {
        "conditional_alignment_weight": 0.0,
        "js_relevance_weight": 0.0,
        "residual_coefficient": 0.25,
        "routing_cardinality": 14,
        "routing_temperature": 1.0,
        "source_only_warmup_iterations": 0,
        "training_iterations": 1000,
        "optimizer_seed": 42,
        "target_bank_history_retention": 0.1,
        "source_bank_history_retention": 0.5,
        "corrected_cluster_centres": True,
    }
    if any(frozen.get(key) != value for key, value in required.items()):
        raise RuntimeError("same-model SEED design differs from executable constants")
    if tuple(frozen["checkpoint_iterations"]) != CHECKPOINTS:
        raise RuntimeError("same-model SEED checkpoint lattice mismatch")
    if design.get("separate_parent_probability_used") is not False:
        raise RuntimeError("same-model SEED design permits a parent probability")
    return design


def evaluate(
    probability_path: Path,
    vault: shared.TargetLabelVault,
    dataset: str,
    session: int,
    subject: int,
    phase: str,
) -> dict[str, Any]:
    with np.load(probability_path, allow_pickle=False) as artifact:
        checkpoints = artifact["checkpoint_iteration"].astype(int)
        probabilities = artifact["probability_trajectory"].astype(np.float64)
    labels = vault.open(probability_path).argmax(axis=1).astype(np.int16)
    rows = [classification_metrics(labels, matrix) for matrix in probabilities]
    best_index = min(range(len(rows)), key=lambda index: (-float(rows[index]["accuracy"]), index))
    return {
        "study": STUDY,
        "tag": TAGS[(dataset, phase)],
        "phase": phase,
        "model": "drsr",
        "dataset": dataset,
        "session": session,
        "target_subject": subject,
        "fixed_final_checkpoint": int(checkpoints[-1]),
        "fixed_final_metrics": rows[-1],
        "target_assisted_checkpoint": int(checkpoints[best_index]),
        "target_assisted_selection_rule": "earliest checkpoint with maximum target accuracy",
        "target_assisted_metrics": rows[best_index],
        "checkpoint_metrics": [
            {"checkpoint": int(checkpoint), **metrics}
            for checkpoint, metrics in zip(checkpoints, rows, strict=True)
        ],
        "target_labels_used_for_training_or_fixed_final_selection": False,
        "target_labels_opened_after_probability_commit": True,
        "target_label_sha256": shared.sha256_array(labels),
    }


def run_fold(dataset: str, session: int, subject: int, phase: str) -> Path:
    design = load_design()
    if dataset not in DATASET_CONTRACT or session not in (1, 2, 3) or subject not in range(1, 16):
        raise ValueError((dataset, session, subject))
    max_iter = 50 if phase == "smoke" else 1000
    checkpoints = (50,) if phase == "smoke" else CHECKPOINTS
    tag = TAGS[(dataset, phase)]
    fold_dir = HERE / "results" / STUDY / tag / dataset / f"session_{session}" / f"target_subject_{subject:02d}"
    if fold_dir.exists():
        raise RuntimeError(f"fresh fold output already exists: {fold_dir}")
    fold_dir.mkdir(parents=True, exist_ok=False)

    setup_seed(42)
    args = build_args(dataset, session, "rwhedn", 42, max_iter)
    args.early_stop = 0
    args.log_interval = 50
    args.balance_source_classes = False
    loaded = get_dataset(args)
    data = np.asarray(loaded["data"], dtype=np.float32)
    labels = np.asarray(loaded["labels"], dtype=np.float32)
    groups = np.asarray(loaded["groups"], dtype=np.int16)
    validate_dataset(dataset, session, data, labels, groups)
    target_mask = groups[:, 0] == subject
    source_mask = ~target_mask
    target_data = np.ascontiguousarray(data[target_mask])
    target_groups = np.ascontiguousarray(groups[target_mask])
    vault = shared.TargetLabelVault(labels[target_mask])
    source_subjects = np.unique(groups[source_mask, 0]).astype(np.int16)
    target_zeros = np.zeros((len(target_data), DATASET_CONTRACT[dataset]["num_classes"]), dtype=np.float32)

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
    if len(source_subjects) != 14 or np.any(target_loader.dataset.d2.numpy()):
        raise RuntimeError("same-model SEED label-isolated LOSO contract failed")
    trainer = shared.build_trainer("drsr", args)
    arrays, training_log, training_seconds, peak_mib = shared.train_model(
        "drsr", trainer, source_loader, target_loader, target_data, source_subjects, checkpoints
    )
    route = arrays.pop("probability_trajectory").astype(np.float64)
    classifier = arrays["classifier_probability_trajectory"].astype(np.float64)
    if classifier.shape != route.shape:
        raise RuntimeError("same-model SEED probability lattices differ")
    pipeline = 0.75 * classifier + 0.25 * route
    pipeline /= np.clip(pipeline.sum(axis=2, keepdims=True), 1e-12, None)
    arrays["route_probability_trajectory"] = route.astype(np.float32)
    arrays["probability_trajectory"] = pipeline.astype(np.float32)

    probability_path = fold_dir / "probabilities.npz"
    trial = target_groups[:, 1].astype(np.int16)
    shared.atomic_npz(
        probability_path,
        **arrays,
        target_subject=np.full(len(target_data), subject, dtype=np.int16),
        target_trial=trial,
        target_window=within_trial_index(trial).astype(np.int32),
        target_session=np.full(len(target_data), session, dtype=np.int16),
    )
    log_path = fold_dir / "training_log.csv"
    np.savetxt(
        log_path,
        training_log,
        delimiter=",",
        header="iteration,classification,transfer,consistency,source_cluster,target_cluster,easy_source_index,hard_source_index",
        comments="",
    )
    evaluation_path = fold_dir / "evaluation.json"
    shared.atomic_json(evaluation_path, evaluate(probability_path, vault, dataset, session, subject, phase))
    manifest = {
        "study": STUDY,
        "tag": tag,
        "phase": phase,
        "model": "drsr",
        "dataset": dataset,
        "dataset_label": DATASET_CONTRACT[dataset]["label"],
        "session": session,
        "target_subject": subject,
        "target_sample_count": int(len(target_data)),
        "source_subject_count": 14,
        "source_subjects": source_subjects.astype(int).tolist(),
        "seed": 42,
        "max_iter": max_iter,
        "batch_size": int(args.batch_size),
        "feature_dim": 310,
        "num_classes": int(args.num_classes),
        "dbscan": {
            "eps": float(loader_builder.best_cluster_params["eps"]),
            "min_samples": int(loader_builder.best_cluster_params["min_samples"]),
        },
        "training_seconds": float(training_seconds),
        "peak_cuda_mib": peak_mib,
        "fixed_design_sha256": sha256_file(DESIGN_PATH),
        "code_sha256": code_sha256(),
        "target_labels_used_for_training_or_fixed_final_selection": False,
        "separate_parent_probability_used": False,
        "artifacts": {
            "probabilities.npz": sha256_file(probability_path),
            "training_log.csv": sha256_file(log_path),
            "evaluation.json": sha256_file(evaluation_path),
        },
        "frozen_configuration": design["frozen_configuration"],
    }
    manifest_path = fold_dir / "fold_manifest.json"
    shared.atomic_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("seed3", "seed4"), required=True)
    parser.add_argument("--session", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--subject", type=int, choices=range(1, 16), required=True)
    parser.add_argument("--phase", choices=("smoke", "production"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(run_fold(args.dataset, args.session, args.subject, args.phase))


if __name__ == "__main__":
    main()
