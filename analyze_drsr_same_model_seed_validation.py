"""Verify and summarize the corrected same-model SEED/SEED-IV production run."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

import drsr_same_model_seed_validation as experiment
from data_utils._get_dataset import get_dataset
from seed_series_upstream_oracle import build_args, validate_dataset
from upstream_v2_residual import classification_metrics


HERE = Path(__file__).resolve().parent
ANALYSIS_ROOT = HERE / "results" / experiment.STUDY / "analysis_r1_20260910"
METRIC_TOLERANCE = 1e-10
RECONSTRUCTION_TOLERANCE = 5e-7


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def summarize(rows: list[dict[str, Any]], prefix: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {"fold_count": len(rows)}
    for metric in ("accuracy", "uar", "macro_f1", "ece"):
        values = np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows])
        result[metric] = float(values.mean())
        result[f"{metric}_sd"] = float(values.std(ddof=1))
    return result


def main() -> None:
    if ANALYSIS_ROOT.exists():
        raise RuntimeError(f"fresh analysis root already exists: {ANALYSIS_ROOT}")
    code_hashes: set[str] = set()
    design_hashes: set[str] = set()
    rows: list[dict[str, Any]] = []
    verified_artifacts = 0
    max_route_diff = 0.0
    max_pipeline_diff = 0.0
    max_metric_diff = 0.0
    dataset_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for dataset in ("seed3", "seed4"):
        tag = experiment.TAGS[(dataset, "production")]
        for session in (1, 2, 3):
            args = build_args(dataset, session, "rwhedn", 42, 1000)
            loaded = get_dataset(args)
            data = np.asarray(loaded["data"], dtype=np.float32)
            labels = np.asarray(loaded["labels"], dtype=np.float32)
            groups = np.asarray(loaded["groups"], dtype=np.int16)
            validate_dataset(dataset, session, data, labels, groups)
            dataset_cache[(dataset, session)] = (labels, groups)
            for subject in range(1, 16):
                fold = HERE / "results" / experiment.STUDY / tag / dataset / f"session_{session}" / f"target_subject_{subject:02d}"
                manifest_path = fold / "fold_manifest.json"
                evaluation_path = fold / "evaluation.json"
                probability_path = fold / "probabilities.npz"
                if not all(path.is_file() for path in (manifest_path, evaluation_path, probability_path, fold / "training_log.csv")):
                    raise RuntimeError(f"incomplete same-model fold: {fold}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
                identity = (manifest.get("dataset"), manifest.get("session"), manifest.get("target_subject"))
                if identity != (dataset, session, subject):
                    raise RuntimeError(f"fold identity mismatch: {fold}")
                if manifest.get("tag") != tag or manifest.get("study") != experiment.STUDY:
                    raise RuntimeError(f"fold tag/study mismatch: {fold}")
                if manifest.get("target_labels_used_for_training_or_fixed_final_selection") is not False:
                    raise RuntimeError(f"label-isolation mismatch: {fold}")
                if manifest.get("separate_parent_probability_used") is not False:
                    raise RuntimeError(f"parent-probability mismatch: {fold}")
                code_hashes.add(manifest["code_sha256"])
                design_hashes.add(manifest["fixed_design_sha256"])
                for name, expected in manifest["artifacts"].items():
                    path = fold / name
                    if experiment.sha256_file(path) != expected:
                        raise RuntimeError(f"artifact hash mismatch: {path}")
                    verified_artifacts += 1
                with np.load(probability_path, allow_pickle=False) as artifact:
                    checkpoints = artifact["checkpoint_iteration"].astype(int)
                    pipeline = artifact["probability_trajectory"].astype(np.float64)
                    classifier = artifact["classifier_probability_trajectory"].astype(np.float64)
                    route = artifact["route_probability_trajectory"].astype(np.float64)
                    selected = artifact["selected_source_probability_trajectory"].astype(np.float64)
                    weights = artifact["selected_source_weight"].astype(np.float64)
                    recorded_subject = artifact["target_subject"].astype(int)
                if tuple(checkpoints) != experiment.CHECKPOINTS or pipeline.shape[0] != 20:
                    raise RuntimeError(f"checkpoint lattice mismatch: {fold}")
                if selected.shape[:2] != (20, 14) or weights.shape != (20, 14):
                    raise RuntimeError(f"source route lattice mismatch: {fold}")
                if not np.all(recorded_subject == subject):
                    raise RuntimeError(f"target-subject artifact mismatch: {fold}")
                rebuilt_route = np.einsum("ts,tsnc->tnc", weights, selected, optimize=True)
                rebuilt_route /= np.clip(rebuilt_route.sum(axis=2, keepdims=True), 1e-12, None)
                route_diff = float(np.max(np.abs(rebuilt_route - route)))
                rebuilt_pipeline = 0.75 * classifier + 0.25 * route
                rebuilt_pipeline /= np.clip(rebuilt_pipeline.sum(axis=2, keepdims=True), 1e-12, None)
                pipeline_diff = float(np.max(np.abs(rebuilt_pipeline - pipeline)))
                max_route_diff = max(max_route_diff, route_diff)
                max_pipeline_diff = max(max_pipeline_diff, pipeline_diff)
                if route_diff > RECONSTRUCTION_TOLERANCE or pipeline_diff > RECONSTRUCTION_TOLERANCE:
                    raise RuntimeError(f"probability reconstruction mismatch: {fold}")

                labels, groups = dataset_cache[(dataset, session)]
                y_true = labels[groups[:, 0] == subject].argmax(axis=1).astype(np.int16)
                if pipeline.shape[1] != len(y_true):
                    raise RuntimeError(f"target sample lattice mismatch: {fold}")
                if experiment.shared.sha256_array(y_true) != evaluation["target_label_sha256"]:
                    raise RuntimeError(f"target-label hash mismatch: {fold}")
                metrics = [classification_metrics(y_true, matrix) for matrix in pipeline]
                best = min(range(len(metrics)), key=lambda index: (-float(metrics[index]["accuracy"]), index))
                if int(evaluation["target_assisted_checkpoint"]) != int(checkpoints[best]):
                    raise RuntimeError(f"assisted checkpoint mismatch: {fold}")
                for expected, actual in (
                    (evaluation["fixed_final_metrics"], metrics[-1]),
                    (evaluation["target_assisted_metrics"], metrics[best]),
                ):
                    for metric in ("accuracy", "uar", "macro_f1", "ece_equal_mass_10"):
                        difference = abs(float(expected[metric]) - float(actual[metric]))
                        max_metric_diff = max(max_metric_diff, difference)
                        if difference > METRIC_TOLERANCE:
                            raise RuntimeError(f"metric reconstruction mismatch: {fold}:{metric}")
                rows.append({
                    "dataset": dataset,
                    "session": session,
                    "subject": subject,
                    "fixed_checkpoint": int(checkpoints[-1]),
                    "assisted_checkpoint": int(checkpoints[best]),
                    "fixed_accuracy": metrics[-1]["accuracy"],
                    "fixed_uar": metrics[-1]["uar"],
                    "fixed_macro_f1": metrics[-1]["macro_f1"],
                    "fixed_ece": metrics[-1]["ece_equal_mass_10"],
                    "assisted_accuracy": metrics[best]["accuracy"],
                    "assisted_uar": metrics[best]["uar"],
                    "assisted_macro_f1": metrics[best]["macro_f1"],
                    "assisted_ece": metrics[best]["ece_equal_mass_10"],
                })
    if len(rows) != 90 or verified_artifacts != 270:
        raise RuntimeError("final same-model fold/artifact count mismatch")
    if code_hashes != {experiment.code_sha256()} or design_hashes != {experiment.sha256_file(experiment.DESIGN_PATH)}:
        raise RuntimeError("final same-model code/design provenance mismatch")

    csv_path = ANALYSIS_ROOT / "fold_metrics.csv"
    atomic_csv(csv_path, rows)
    groups_out: dict[str, Any] = {}
    for dataset in ("seed3", "seed4"):
        subset = [row for row in rows if row["dataset"] == dataset]
        groups_out[dataset] = {
            "overall": {"fixed_final": summarize(subset, "fixed"), "target_assisted": summarize(subset, "assisted")},
            "sessions": {
                str(session): {
                    "fixed_final": summarize([row for row in subset if row["session"] == session], "fixed"),
                    "target_assisted": summarize([row for row in subset if row["session"] == session], "assisted"),
                }
                for session in (1, 2, 3)
            },
        }
    summary = {
        "status": "passed",
        "study": experiment.STUDY,
        "fold_count": 90,
        "verified_artifact_hashes": verified_artifacts,
        "code_sha256": next(iter(code_hashes)),
        "design_sha256": next(iter(design_hashes)),
        "max_route_reconstruction_abs": max_route_diff,
        "max_pipeline_reconstruction_abs": max_pipeline_diff,
        "max_metric_reconstruction_abs": max_metric_diff,
        "fold_metrics_sha256": experiment.sha256_file(csv_path),
        "datasets": groups_out,
    }
    experiment.shared.atomic_json(ANALYSIS_ROOT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
