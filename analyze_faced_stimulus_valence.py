"""Post-hoc stimulus-valence sensitivity for the frozen FACED endpoint.

This analysis does not retrain DRSR or select a checkpoint.  It collapses the
committed iteration-1000 nine-class probabilities into non-positive stimulus
classes (anger, disgust, fear, sadness, neutral) and positive stimulus classes
(amusement, inspiration, joy, tenderness).  It is contextual sensitivity
evidence, not a reproduction of self-report valence protocols such as EmT.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
STUDY_ROOT = (
    HERE
    / "results"
    / "faced_external_validation"
    / "faced_external_same_model_production_r3_20260909"
)
MODEL_ROOT = STUDY_ROOT / "drsr"
ANALYSIS_ROOT = STUDY_ROOT / "analysis"
SUMMARY_PATH = ANALYSIS_ROOT / "stimulus_valence_sensitivity.json"
PARTICIPANT_PATH = ANALYSIS_ROOT / "stimulus_valence_participant_metrics.csv"

NUM_SUBJECTS = 123
NUM_VIDEOS = 28
NUM_WINDOWS = 30
NUM_CLASSES = 9
FIXED_CHECKPOINT = 1000
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 20260915

VIDEO_LABELS = np.asarray(
    [0] * 3
    + [1] * 3
    + [2] * 3
    + [3] * 3
    + [4] * 4
    + [5] * 3
    + [6] * 3
    + [7] * 3
    + [8] * 3,
    dtype=np.int16,
)
WINDOW_BINARY_LABELS = np.repeat((VIDEO_LABELS >= 5).astype(np.int16), NUM_WINDOWS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def binary_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability.argmax(axis=1).astype(np.int16)
    recalls: list[float] = []
    f1s: list[float] = []
    for label in (0, 1):
        true_positive = int(np.sum((labels == label) & (prediction == label)))
        false_positive = int(np.sum((labels != label) & (prediction == label)))
        false_negative = int(np.sum((labels == label) & (prediction != label)))
        recalls.append(true_positive / (true_positive + false_negative))
        denominator = 2 * true_positive + false_positive + false_negative
        f1s.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "accuracy": float(np.mean(labels == prediction)),
        "uar": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "positive_class_f1": float(f1s[1]),
    }


def participant_interval(values: np.ndarray, draw_indices: np.ndarray) -> list[float]:
    bootstrap_means = values[draw_indices].mean(axis=1)
    return np.quantile(bootstrap_means, [0.025, 0.975]).astype(float).tolist()


def verify_and_measure(subject: int) -> tuple[dict[str, Any], str]:
    fold = MODEL_ROOT / f"target_subject_{subject:03d}"
    manifest_path = fold / "fold_manifest.json"
    probability_path = fold / "probabilities.npz"
    evaluation_path = fold / "evaluation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if (
        manifest.get("target_subject") != subject
        or manifest.get("num_classes") != NUM_CLASSES
        or manifest.get("target_labels_used_for_training_or_selection") is not False
        or evaluation.get("target_labels_used_for_training_or_fixed_final_selection") is not False
        or evaluation.get("fixed_final_checkpoint") != FIXED_CHECKPOINT
    ):
        raise RuntimeError(f"FACED identity or endpoint failure: {fold}")
    if sha256_file(probability_path) != manifest["artifacts"]["probabilities.npz"]:
        raise RuntimeError(f"FACED probability hash mismatch: {probability_path}")

    with np.load(probability_path, allow_pickle=False) as artifact:
        checkpoints = artifact["checkpoint_iteration"].astype(int)
        trajectory = artifact["probability_trajectory"].astype(np.float64)
    if trajectory.shape[1:] != (NUM_VIDEOS * NUM_WINDOWS, NUM_CLASSES):
        raise RuntimeError(f"FACED probability lattice failure: {probability_path}")
    fixed_indices = np.flatnonzero(checkpoints == FIXED_CHECKPOINT)
    if len(fixed_indices) != 1:
        raise RuntimeError(f"FACED fixed checkpoint failure: {probability_path}")
    probability = trajectory[int(fixed_indices[0])]
    binary_probability = np.stack(
        (probability[:, :5].sum(axis=1), probability[:, 5:].sum(axis=1)),
        axis=1,
    )
    binary_probability /= binary_probability.sum(axis=1, keepdims=True)
    return (
        {"participant": subject, **binary_metrics(WINDOW_BINARY_LABELS, binary_probability)},
        str(manifest["code_sha256"]),
    )


def main() -> None:
    if SUMMARY_PATH.exists() or PARTICIPANT_PATH.exists():
        raise RuntimeError("fresh FACED stimulus-valence analysis output already exists")
    rows: list[dict[str, Any]] = []
    code_hashes: set[str] = set()
    for subject in range(1, NUM_SUBJECTS + 1):
        row, code_hash = verify_and_measure(subject)
        rows.append(row)
        code_hashes.add(code_hash)
    if len(code_hashes) != 1:
        raise RuntimeError(f"FACED code provenance is not uniform: {sorted(code_hashes)}")

    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = PARTICIPANT_PATH.with_suffix(f".csv.{os.getpid()}.tmp")
    with temporary.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("participant", "accuracy", "uar", "macro_f1", "positive_class_f1"),
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, PARTICIPANT_PATH)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw_indices = rng.integers(
        0, NUM_SUBJECTS, size=(BOOTSTRAP_DRAWS, NUM_SUBJECTS), dtype=np.int16
    )
    estimates: dict[str, Any] = {}
    for metric in ("accuracy", "uar", "macro_f1", "positive_class_f1"):
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        estimates[metric] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "participant_bootstrap_95_ci": participant_interval(values, draw_indices),
        }

    payload = {
        "study": "faced_stimulus_valence_sensitivity",
        "status": "passed",
        "analysis_status": "post_hoc_contextual_sensitivity",
        "participant_count": NUM_SUBJECTS,
        "fixed_checkpoint": FIXED_CHECKPOINT,
        "probability_source": "frozen nine-class iteration-1000 DRSR probabilities",
        "class_mapping": {
            "non_positive": ["anger", "disgust", "fear", "sadness", "neutral"],
            "positive": ["amusement", "inspiration", "joy", "tenderness"],
        },
        "estimand": "participant-mean window-level metric after probability collapse",
        "estimates": estimates,
        "bootstrap": {
            "unit": "participant",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
        },
        "integrity": {
            "folds_verified": NUM_SUBJECTS,
            "source_code_sha256": next(iter(code_hashes)),
            "analysis_code_sha256": sha256_file(Path(__file__)),
            "participant_metrics_sha256": sha256_file(PARTICIPANT_PATH),
            "target_labels_used_for_training_or_checkpoint_selection": False,
        },
        "interpretation_constraints": [
            "The native nine-class FACED endpoint remains primary.",
            "This probability collapse was designed after the primary FACED results were known.",
            "Stimulus categories proxy valence; EmT thresholds participant self-report ratings.",
            "The analysis does not retrain a binary classifier or reproduce EmT preprocessing, folds, or metrics.",
            "Numerical proximity to EmT is contextual and does not establish superiority or equivalence.",
        ],
    }
    atomic_json(SUMMARY_PATH, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
