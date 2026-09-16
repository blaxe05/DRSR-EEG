"""Audit and summarize the completed frozen FACED evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

import faced_external_validation as experiment


HERE = Path(__file__).resolve().parent
ROOT = HERE / "results" / experiment.STUDY / experiment.PRODUCTION_TAG
ANALYSIS_ROOT = ROOT / "analysis"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def paired_interval(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(20260909)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975]).astype(float).tolist()


def sign_flip_p(values: np.ndarray) -> float:
    rng = np.random.default_rng(20260909)
    observed = abs(float(values.mean()))
    exceed = 0
    total = 100000
    for _ in range(total // 1000):
        signs = rng.choice((-1.0, 1.0), size=(1000, len(values)))
        exceed += int(np.sum(np.abs((signs * values).mean(axis=1)) >= observed))
    return float((exceed + 1) / (total + 1))


def verify_fold(model: str, subject: int) -> tuple[dict[str, Any], dict[str, Any]]:
    fold = ROOT / model / f"target_subject_{subject:03d}"
    manifest_path = fold / "fold_manifest.json"
    evaluation_path = fold / "evaluation.json"
    probability_path = fold / "probabilities.npz"
    log_path = fold / "training_log.csv"
    required = (manifest_path, evaluation_path, probability_path, log_path)
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"incomplete FACED fold: {fold}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if (
        manifest.get("model") != model
        or manifest.get("target_subject") != subject
        or manifest.get("source_subject_count") != 122
        or manifest.get("target_labels_used_for_training_or_selection") is not False
        or evaluation.get("target_labels_used_for_training_or_fixed_final_selection") is not False
    ):
        raise RuntimeError(f"FACED identity or label-boundary failure: {fold}")
    for name, expected in manifest["artifacts"].items():
        if experiment.sha256_file(fold / name) != expected:
            raise RuntimeError(f"FACED artifact hash mismatch: {fold / name}")
    with np.load(probability_path, allow_pickle=False) as artifact:
        checkpoints = artifact["checkpoint_iteration"]
        probability = artifact["probability_trajectory"]
        if probability.shape != (20, 840, 9) or not np.array_equal(
            checkpoints, np.asarray(experiment.CHECKPOINTS)
        ):
            raise RuntimeError(f"FACED probability lattice mismatch: {fold}")
        if not np.isfinite(probability).all():
            raise RuntimeError(f"non-finite FACED probabilities: {fold}")
        if model == "drsr":
            selected = artifact["selected_source_probability_trajectory"]
            weights = artifact["selected_source_weight"]
            route = artifact["route_probability_trajectory"]
            classifier = artifact["classifier_probability_trajectory"]
            reconstructed = np.einsum("ts,tsnc->tnc", weights, selected, optimize=True)
            reconstructed /= reconstructed.sum(axis=2, keepdims=True)
            route_difference = float(np.max(np.abs(reconstructed - route)))
            pipeline = 0.75 * classifier + 0.25 * route
            pipeline /= pipeline.sum(axis=2, keepdims=True)
            pipeline_difference = float(np.max(np.abs(pipeline - probability)))
            if route_difference > 5e-7 or pipeline_difference > 5e-7:
                raise RuntimeError(f"FACED reconstruction failure: {fold}")
            manifest["route_reconstruction_max_abs"] = route_difference
            manifest["pipeline_reconstruction_max_abs"] = pipeline_difference
    return manifest, evaluation


def aggregate(rows: list[dict[str, Any]], model: str, endpoint: str) -> dict[str, Any]:
    selected = [row for row in rows if row["model"] == model]
    prefix = "fixed_final" if endpoint == "fixed_final" else "target_assisted"
    metrics = [row[f"{prefix}_metrics"] for row in selected]
    result: dict[str, Any] = {"participant_count": len(metrics)}
    for name in ("accuracy", "uar", "macro_f1", "ece_equal_mass_10"):
        values = np.asarray([float(metric[name]) for metric in metrics])
        result[name] = float(values.mean())
        result[f"{name}_sd"] = float(values.std(ddof=1))
    return result


def main() -> None:
    if ANALYSIS_ROOT.exists():
        raise RuntimeError(f"fresh FACED analysis output already exists: {ANALYSIS_ROOT}")
    rows: list[dict[str, Any]] = []
    maxima = {"route": 0.0, "pipeline": 0.0}
    code_hashes: set[str] = set()
    for model in ("drsr",):
        for subject in range(1, 124):
            manifest, evaluation = verify_fold(model, subject)
            code_hashes.add(str(manifest["code_sha256"]))
            maxima["route"] = max(
                maxima["route"], float(manifest.get("route_reconstruction_max_abs", 0.0))
            )
            maxima["pipeline"] = max(
                maxima["pipeline"], float(manifest.get("pipeline_reconstruction_max_abs", 0.0))
            )
            rows.append(
                {
                    "model": model,
                    "subject": subject,
                    "fixed_final_checkpoint": evaluation["fixed_final_checkpoint"],
                    "target_assisted_checkpoint": evaluation["target_assisted_checkpoint"],
                    "fixed_final_metrics": evaluation["fixed_final_metrics"],
                    "target_assisted_metrics": evaluation["target_assisted_metrics"],
                }
            )
    if len(code_hashes) != 1:
        raise RuntimeError(f"FACED code provenance is not uniform: {sorted(code_hashes)}")

    summary: dict[str, Any] = {
        "study": experiment.STUDY,
        "tag": experiment.PRODUCTION_TAG,
        "status": "passed",
        "fold_count": 123,
        "participant_count": 123,
        "code_sha256": next(iter(code_hashes)),
        "max_route_reconstruction_abs": maxima["route"],
        "max_pipeline_reconstruction_abs": maxima["pipeline"],
        "models": {},
    }
    for model in ("drsr",):
        summary["models"][model] = {
            endpoint: aggregate(rows, model, endpoint)
            for endpoint in ("fixed_final", "target_assisted")
        }

    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=False)
    csv_path = ANALYSIS_ROOT / "participant_metrics.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "subject", "fixed_final_checkpoint", "target_assisted_checkpoint",
            "fixed_accuracy", "fixed_uar", "fixed_macro_f1", "fixed_ece",
            "assisted_accuracy", "assisted_uar", "assisted_macro_f1", "assisted_ece",
        ])
        for row in rows:
            fixed = row["fixed_final_metrics"]
            assisted = row["target_assisted_metrics"]
            writer.writerow([
                row["model"], row["subject"], row["fixed_final_checkpoint"],
                row["target_assisted_checkpoint"], fixed["accuracy"], fixed["uar"],
                fixed["macro_f1"], fixed["ece_equal_mass_10"], assisted["accuracy"],
                assisted["uar"], assisted["macro_f1"], assisted["ece_equal_mass_10"],
            ])
    summary["participant_metrics_sha256"] = experiment.sha256_file(csv_path)
    atomic_json(ANALYSIS_ROOT / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
