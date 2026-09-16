"""Offline baseline-recoverable residual mixture over immutable V2 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
STUDY = "upstream_v2_residual"
TAGS = {"smoke": "smoke_upstream_v2_residual_v1", "production": "paper_upstream_v2_residual_v1"}
BASELINE_TAGS = {
    "smoke": "smoke_upstream_v2_residual_baseline_v1",
    "production": "paper_upstream_v2_residual_baseline_v1",
}
SPARSE_TAGS = {
    "smoke": "smoke_upstream_v2_sparse_topk_v1",
    "production": "paper_upstream_v2_sparse_topk_v1",
}
TEMPERATURE_TAGS = {
    "smoke": "smoke_upstream_v2_temperature_v1",
    "production": "paper_upstream_v2_temperature_v1",
}
PARITY_TAG = "paper_upstream_v2_baseline_parity_label_isolated_v1"
GATES = (0.00, 0.25, 0.50, 0.75, 1.00)
TOL = 1e-10
EPS = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise RuntimeError("invalid probability matrix")
    if (values < -EPS).any():
        raise RuntimeError("negative probability")
    values = np.clip(values, 0.0, None)
    sums = values.sum(axis=1, keepdims=True)
    if (sums <= EPS).any():
        raise RuntimeError("zero probability row")
    return values / sums


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = normalize(probabilities)
    if labels.shape != (probabilities.shape[0],):
        raise RuntimeError("label/probability shape mismatch")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    order = np.argsort(confidence, kind="stable")
    ece = 0.0
    for indices in np.array_split(order, 10):
        if len(indices):
            ece += len(indices) / len(labels) * abs(
                float(confidence[indices].mean())
                - float((predictions[indices] == labels[indices]).mean())
            )
    classes = np.arange(probabilities.shape[1])
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for actual, predicted in zip(labels, predictions, strict=True):
        confusion[int(actual), int(predicted)] += 1
    recalls: list[float] = []
    f1s: list[float] = []
    for class_index in classes:
        true_positive = int(confusion[class_index, class_index])
        false_negative = int(confusion[class_index, :].sum()) - true_positive
        false_positive = int(confusion[:, class_index].sum()) - true_positive
        recalls.append(true_positive / max(1, true_positive + false_negative))
        f1s.append(2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative))
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "uar": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "classwise_f1": [float(value) for value in f1s],
        "ece_equal_mass_10": float(ece),
        "confusion_matrix": confusion.tolist(),
    }


def tail_metrics(values: list[float]) -> tuple[float, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    count = max(1, int(math.ceil(0.2 * len(ordered))))
    return float(ordered[:count].mean()), float(np.quantile(ordered, 0.1))


def route_probability(trajectory_path: Path, selected_top_k: int, temperature: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(trajectory_path, allow_pickle=False) as archive:
        per_source = archive["per_source_probability_trajectory"][-1].astype(np.float64)
        relevance = archive["relevance_trajectory"][-1].astype(np.float64)
        source_subjects = archive["source_subject"].astype(np.int64)
        identities = {
            "subject": archive["target_subject"].astype(np.int64),
            "session": archive["target_session"].astype(np.int64),
            "trial": archive["target_trial"].astype(np.int64),
            "sample": archive["target_sample_within_trial"].astype(np.int64),
            "index": archive["target_original_dataset_index"].astype(np.int64),
            "retained": archive["retained_mask"].astype(bool),
        }
    if per_source.shape[0] != len(relevance) or len(relevance) != len(source_subjects):
        raise RuntimeError(f"route source-shape mismatch: {trajectory_path}")
    order = np.lexsort((source_subjects, -relevance))
    keep = order[: min(int(selected_top_k), len(source_subjects))]
    weights = np.zeros(len(source_subjects), dtype=np.float64)
    values = np.power(np.clip(relevance[keep], EPS, None), 1.0 / float(temperature))
    weights[keep] = values / values.sum()
    mixture = np.einsum("s,snc->nc", weights, per_source, optimize=True)
    return normalize(mixture), identities


def baseline_probability(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        probabilities = archive["all_last_batched_probs"].astype(np.float64)
        identities = {
            "subject": archive["all_subject"].astype(np.int64),
            "session": archive["all_session"].astype(np.int64),
            "trial": archive["all_trial"].astype(np.int64),
            "sample": archive["all_sample_within_trial"].astype(np.int64),
            "index": archive["all_original_dataset_index"].astype(np.int64),
            "retained": archive["retained_mask"].astype(bool),
        }
    return normalize(probabilities), identities


def assert_identity(left: dict[str, np.ndarray], right: dict[str, np.ndarray], context: str) -> None:
    if set(left) != set(right):
        raise RuntimeError(f"identity schema mismatch: {context}")
    for key in sorted(left):
        if not np.array_equal(left[key], right[key]):
            raise RuntimeError(f"sample identity mismatch key={key}: {context}")


def read_labels(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return archive[key].astype(np.int64), archive["target_original_dataset_index" if key == "y_true" else "all_original_dataset_index"].astype(np.int64)


def select_gate(inner_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], float]:
    rows: list[dict[str, object]] = []
    for gate in GATES:
        values = [row["candidates"][str(gate)] for row in inner_rows]
        uars = [float(value["uar"]) for value in values]
        cvar20, p10 = tail_metrics(uars)
        rows.append(
            {
                "gate": gate,
                "mean_uar": float(np.mean(uars)),
                "mean_accuracy": float(np.mean([float(value["accuracy"]) for value in values])),
                "cvar20_uar": cvar20,
                "p10_uar": p10,
                "mean_ece": float(np.mean([float(value["ece_equal_mass_10"]) for value in values])),
            }
        )
    baseline = next(row for row in rows if row["gate"] == 0.0)
    eligible_nonzero: list[dict[str, object]] = []
    for row in rows:
        if row["gate"] == 0.0:
            row["eligible"] = True
            row["eligibility_reason"] = "baseline"
            continue
        eligible = (
            float(row["mean_uar"]) > float(baseline["mean_uar"]) + TOL
            and float(row["mean_accuracy"]) >= float(baseline["mean_accuracy"]) - TOL
            and float(row["cvar20_uar"]) >= float(baseline["cvar20_uar"]) - TOL
            and float(row["p10_uar"]) >= float(baseline["p10_uar"]) - TOL
            and float(row["mean_ece"]) <= float(baseline["mean_ece"]) + TOL
        )
        row["eligible"] = eligible
        row["eligibility_reason"] = "strict_uar_and_guardrails" if eligible else "ineligible"
        if eligible:
            eligible_nonzero.append(row)
    if not eligible_nonzero:
        return rows, 0.0
    winner = max(
        eligible_nonzero,
        key=lambda row: (
            float(row["mean_uar"]),
            float(row["mean_accuracy"]),
            float(row["cvar20_uar"]),
            float(row["p10_uar"]),
            -float(row["mean_ece"]),
            -float(row["gate"]),
        ),
    )
    return rows, float(winner["gate"])


def fold(phase: str, dataset: str, session: int, outer_subject: int) -> dict[str, object]:
    output = HERE / "results" / STUDY / TAGS[phase] / dataset / f"session_{session}" / f"target_subject_{outer_subject:02d}"
    output.mkdir(parents=True, exist_ok=False)
    baseline_root = HERE / "results" / "upstream_v2_residual_baseline" / BASELINE_TAGS[phase] / dataset / f"session_{session}" / f"outer_subject_{outer_subject:02d}"
    sparse_root = HERE / "results" / "upstream_v2_sparse_topk" / SPARSE_TAGS[phase] / dataset / f"session_{session}" / f"target_subject_{outer_subject:02d}"
    temperature_path = HERE / "results" / "upstream_v2_temperature" / TEMPERATURE_TAGS[phase] / dataset / f"session_{session}" / f"target_subject_{outer_subject:02d}" / "temperature_fold.json"
    parity_root = HERE / "results" / "upstream_v2_baseline_parity" / PARITY_TAG / dataset / f"session_{session}" / "label_isolated" / f"target_subject_{outer_subject:02d}"
    temperature_document = json.loads(temperature_path.read_text(encoding="utf-8"))
    selected_top_k = int(temperature_document["selected_top_k_requested"])
    selected_temperature = float(temperature_document["selected_temperature"])
    sparse_selection_path = sparse_root / "selection.json"
    sparse_selection = json.loads(sparse_selection_path.read_text(encoding="utf-8"))
    if int(sparse_selection["selected_top_k_requested"]) != selected_top_k:
        raise RuntimeError("temperature/sparse selected-top-k mismatch")

    input_hashes: dict[str, str] = {
        "temperature_fold.json": sha256_file(temperature_path),
        "sparse_selection.json": sha256_file(sparse_selection_path),
    }
    inner_rows: list[dict[str, object]] = []
    for pseudo_target in range(1, 16):
        if pseudo_target == outer_subject:
            continue
        baseline_dir = baseline_root / f"inner_target_{pseudo_target:02d}"
        sparse_dir = sparse_root / f"inner_target_{pseudo_target:02d}"
        baseline_path = baseline_dir / "baseline_probabilities.npz"
        baseline_labels_path = baseline_dir / "target_labels.npz"
        route_path = sparse_dir / "trajectories.npz"
        route_labels_path = sparse_dir / "target_labels.npz"
        for name, path in (
            (f"inner_{pseudo_target:02d}/baseline_probabilities.npz", baseline_path),
            (f"inner_{pseudo_target:02d}/baseline_target_labels.npz", baseline_labels_path),
            (f"inner_{pseudo_target:02d}/sparse_trajectories.npz", route_path),
            (f"inner_{pseudo_target:02d}/sparse_target_labels.npz", route_labels_path),
        ):
            input_hashes[name] = sha256_file(path)
        base_probability, base_identity = baseline_probability(baseline_path)
        route_probability_matrix, route_identity = route_probability(route_path, selected_top_k, selected_temperature)
        assert_identity(base_identity, route_identity, f"inner target {pseudo_target}")
        base_labels, base_indices = read_labels(baseline_labels_path, "y_true")
        route_labels, route_indices = read_labels(route_labels_path, "y_true")
        if not np.array_equal(base_indices, base_identity["index"]) or not np.array_equal(route_indices, route_identity["index"]):
            raise RuntimeError(f"inner label identity mismatch target={pseudo_target}")
        if not np.array_equal(base_labels, route_labels):
            raise RuntimeError(f"inner label disagreement target={pseudo_target}")
        candidates = {
            str(gate): classification_metrics(
                base_labels,
                (1.0 - gate) * base_probability + gate * route_probability_matrix,
            )
            for gate in GATES
        }
        inner_rows.append({"pseudo_target": pseudo_target, "candidates": candidates})

    candidate_rows, selected_gate = select_gate(inner_rows)
    selection = {
        "study": STUDY,
        "run_tag": TAGS[phase],
        "phase": phase,
        "dataset": dataset,
        "session": session,
        "outer_subject": outer_subject,
        "gate_candidates": list(GATES),
        "selected_top_k_requested": selected_top_k,
        "selected_temperature": selected_temperature,
        "candidate_rows": candidate_rows,
        "selected_gate": selected_gate,
        "inner": inner_rows,
        "outer_labels_opened": False,
        "input_sha256": input_hashes,
        "committed_at": datetime.now(timezone.utc).isoformat(),
    }
    selection_path = output / "gate_selection.json"
    atomic_json(selection_path, selection)

    outer_base_path = parity_root / "probabilities.npz"
    outer_base_labels_path = parity_root / "target_labels.npz"
    outer_route_path = sparse_root / "outer_fit" / "trajectories.npz"
    outer_route_labels_path = sparse_root / "outer_fit" / "target_labels.npz"
    for name, path in (
        ("outer/baseline_probabilities.npz", outer_base_path),
        ("outer/baseline_target_labels.npz", outer_base_labels_path),
        ("outer/sparse_trajectories.npz", outer_route_path),
        ("outer/sparse_target_labels.npz", outer_route_labels_path),
    ):
        input_hashes[name] = sha256_file(path)
    outer_base, outer_base_identity = baseline_probability(outer_base_path)
    outer_route, outer_route_identity = route_probability(outer_route_path, selected_top_k, selected_temperature)
    assert_identity(outer_base_identity, outer_route_identity, "outer fold")
    selected_probability = normalize((1.0 - selected_gate) * outer_base + selected_gate * outer_route)
    probabilities_path = output / "residual_probabilities.npz"
    atomic_npz(
        probabilities_path,
        baseline_probability=outer_base.astype(np.float32),
        route_probability=outer_route.astype(np.float32),
        selected_probability=selected_probability.astype(np.float32),
        selected_gate=np.asarray([selected_gate], dtype=np.float64),
        subject=outer_base_identity["subject"].astype(np.int16),
        session=outer_base_identity["session"].astype(np.int8),
        trial=outer_base_identity["trial"].astype(np.int16),
        sample_within_trial=outer_base_identity["sample"].astype(np.int32),
        original_dataset_index=outer_base_identity["index"].astype(np.int32),
        retained_mask=outer_base_identity["retained"].astype(bool),
    )

    # Outer labels are opened only after gate selection and mixture probabilities commit.
    outer_labels, outer_label_indices = read_labels(outer_base_labels_path, "all_y_true")
    route_labels, route_label_indices = read_labels(outer_route_labels_path, "y_true")
    if not np.array_equal(outer_label_indices, outer_base_identity["index"]) or not np.array_equal(route_label_indices, outer_route_identity["index"]):
        raise RuntimeError("outer label identity mismatch")
    if not np.array_equal(outer_labels, route_labels):
        raise RuntimeError("outer label disagreement")
    outer_candidates = {
        str(gate): classification_metrics(outer_labels, (1.0 - gate) * outer_base + gate * outer_route)
        for gate in GATES
    }
    outer_result = {
        "study": STUDY,
        "run_tag": TAGS[phase],
        "phase": phase,
        "dataset": dataset,
        "session": session,
        "outer_subject": outer_subject,
        "selected_gate": selected_gate,
        "selected_top_k_requested": selected_top_k,
        "selected_temperature": selected_temperature,
        "outer_candidates": outer_candidates,
        "outer_selected_metrics": outer_candidates[str(selected_gate)],
        "outer_baseline_metrics": outer_candidates["0.0"],
        "gate_selection_sha256": sha256_file(selection_path),
        "residual_probabilities_sha256": sha256_file(probabilities_path),
        "outer_labels_used_for_selection": False,
        "label_vault_transitions": [
            {"sequence": 0, "state": "sealed_through_gate_and_probability_commit"},
            {"sequence": 1, "state": "opened_for_posthoc_evaluation"},
        ],
        "input_sha256": input_hashes,
    }
    outer_result_path = output / "outer_result.json"
    atomic_json(outer_result_path, outer_result)
    ledger = {
        "study": STUDY,
        "run_tag": TAGS[phase],
        "phase": phase,
        "dataset": dataset,
        "session": session,
        "outer_subject": outer_subject,
        "artifact_sha256": {
            "gate_selection.json": sha256_file(selection_path),
            "residual_probabilities.npz": sha256_file(probabilities_path),
            "outer_result.json": sha256_file(outer_result_path),
        },
    }
    atomic_json(output / "fold_ledger.json", ledger)
    return {
        "dataset": dataset,
        "session": session,
        "outer_subject": outer_subject,
        "selected_gate": selected_gate,
        "outer_selected_metrics": outer_candidates[str(selected_gate)],
        "delta_uar_vs_baseline": float(outer_candidates[str(selected_gate)]["uar"]) - float(outer_candidates["0.0"]["uar"]),
    }


def expected_folds(phase: str) -> list[tuple[str, int, int]]:
    if phase == "smoke":
        return [("seed3", 1, 1), ("seed4", 1, 1)]
    return [
        (dataset, session, subject)
        for dataset in ("seed3", "seed4")
        for session in (1, 2, 3)
        for subject in range(1, 16)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=TAGS, required=True)
    args = parser.parse_args()
    for identity in expected_folds(args.phase):
        print(json.dumps(fold(args.phase, *identity), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
