"""Technical and statistical audit for the core DRSR routing ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

import core_routing_ablation as study
import upstream_v2_js_relevance as stage_e1


HERE = Path(__file__).resolve().parent
SMOKE_FOLDS = (
    ("seed3", 1, 1),
    ("seed3", 1, 2),
    ("seed4", 1, 14),
    ("seed4", 1, 15),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_folds(phase: str) -> list[tuple[str, int, int]]:
    if phase == "smoke":
        return list(SMOKE_FOLDS)
    return [
        (dataset, session, subject)
        for dataset in ("seed3", "seed4")
        for session in (1, 2, 3)
        for subject in range(1, 16)
    ]


def exact_metric_difference(recorded: dict[str, Any], recomputed: dict[str, Any]) -> float:
    keys = (
        "accuracy", "uar", "macro_f1", "classwise_f1", "brier",
        "ece_equal_mass_10", "confidence_mean", "confusion_matrix",
        "retained_accuracy", "all_window_accuracy", "coverage",
    )
    differences = []
    for key in keys:
        if key in recorded and key in recomputed:
            left = np.asarray(recorded[key], dtype=np.float64)
            right = np.asarray(recomputed[key], dtype=np.float64)
            if left.shape != right.shape:
                raise RuntimeError(f"metric shape mismatch for {key}: {left.shape} != {right.shape}")
            differences.append(float(np.max(np.abs(left - right))))
    if not differences:
        raise RuntimeError("no common metrics available for reconstruction")
    return max(differences)


def audit_ablation_fold(
    run_tag: str, mode: str, dataset: str, session: int, subject: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = study.fold_directory(run_tag, mode, dataset, session, subject)
    launch = {
        "code_sha256": study.code_provenance(),
        "git": study.git_provenance(),
    }
    if not study.validate_complete_fold(
        root, run_tag, mode, dataset, session, subject, launch
    ):
        raise RuntimeError(f"incomplete ablation fold: {root}")
    evaluation = json.loads(
        (root / "outer_fit" / "evaluation.json").read_text(encoding="utf-8")
    )
    result = json.loads((root / "outer_result.json").read_text(encoding="utf-8"))
    with np.load(root / "outer_fit" / "pipeline_probabilities.npz", allow_pickle=False) as archive:
        baseline = archive["baseline_probability"].astype(np.float64)
        route = archive["route_probability"].astype(np.float64)
        selected = archive["selected_probability"].astype(np.float64)
        gate = float(archive["selected_gate"][0])
        sample_subject = archive["target_subject"].astype(np.int16)
        sample_session = archive["target_session"].astype(np.int8)
        sample_trial = archive["target_trial"].astype(np.int16)
        sample_within = archive["target_sample_within_trial"].astype(np.int32)
    reconstructed = stage_e1.normalize((1.0 - gate) * baseline + gate * route)
    reconstruction_diff = float(np.max(np.abs(reconstructed - selected)))
    with np.load(root / "outer_fit" / "target_labels.npz", allow_pickle=False) as archive:
        y_true = archive["y_true"].astype(np.int16)
        label_subject = archive["target_subject"].astype(np.int16)
        label_session = archive["target_session"].astype(np.int8)
        label_trial = archive["target_trial"].astype(np.int16)
        label_within = archive["target_sample_within_trial"].astype(np.int32)
    if not (
        np.array_equal(sample_subject, label_subject)
        and np.array_equal(sample_session, label_session)
        and np.array_equal(sample_trial, label_trial)
        and np.array_equal(sample_within, label_within)
    ):
        raise RuntimeError(f"sample identity mismatch: {root}")
    recomputed = stage_e1.multiclass_metrics(y_true, selected)
    metric_diff = exact_metric_difference(evaluation["selected_metrics"], recomputed)
    if reconstruction_diff > 5e-7 or metric_diff > 1e-10:
        raise RuntimeError(
            f"numerical gate failed at {root}: reconstruction={reconstruction_diff} "
            f"metric={metric_diff}"
        )
    row = {
        "dataset": dataset,
        "session": session,
        "subject": subject,
        "routing_mode": mode,
        **{key: float(value) for key, value in recomputed.items() if np.isscalar(value)},
    }
    technical = {
        "fold": str(root),
        "reconstruction_max_abs_diff": reconstruction_diff,
        "metric_max_abs_diff": metric_diff,
        "verified_hashes": len(
            json.loads((root / "fold_ledger.json").read_text(encoding="utf-8"))["sha256"]
        ),
    }
    return row, technical


def load_parent_row(dataset: str, session: int, subject: int) -> dict[str, Any]:
    parent = study.load_parent_reference(dataset, session, subject)
    metrics = parent["selected_outer_metrics"]
    return {
        "dataset": dataset,
        "session": session,
        "subject": subject,
        "routing_mode": "full",
        **{key: float(value) for key, value in metrics.items() if np.isscalar(value)},
    }


def full_control_difference(run_tag: str, dataset: str, session: int, subject: int) -> float:
    root = study.fold_directory(run_tag, "full", dataset, session, subject)
    parent = study.parent_fold(dataset, session, subject)
    with np.load(root / "outer_fit" / "pipeline_probabilities.npz", allow_pickle=False) as left:
        candidate = left["selected_probability"].astype(np.float64)
    with np.load(parent / "outer_fit" / "pipeline_probabilities.npz", allow_pickle=False) as right:
        reference = right["selected_probability"].astype(np.float64)
    if candidate.shape != reference.shape:
        raise RuntimeError("full-control probability shape differs from E2 parent")
    return float(np.max(np.abs(candidate - reference)))


def participant_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in ("seed3", "seed4"):
        for mode in ("full", *study.PRODUCTION_MODES):
            for subject in range(1, 16):
                selected = [
                    row for row in rows
                    if row["dataset"] == dataset
                    and row["routing_mode"] == mode
                    and row["subject"] == subject
                ]
                if len(selected) != 3:
                    raise RuntimeError((dataset, mode, subject, len(selected)))
                output.append({
                    "dataset": dataset,
                    "routing_mode": mode,
                    "subject": subject,
                    "uar": float(np.mean([row["uar"] for row in selected])),
                    "accuracy": float(np.mean([row["accuracy"] for row in selected])),
                    "macro_f1": float(np.mean([row["macro_f1"] for row in selected])),
                    "brier": float(np.mean([row["brier"] for row in selected])),
                    "ece": float(np.mean([row["ece_equal_mass_10"] for row in selected])),
                })
    return output


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(100000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def summarize(participants: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset_index, dataset in enumerate(("seed3", "seed4")):
        summary[dataset] = {}
        full = np.asarray([
            row["uar"] for row in participants
            if row["dataset"] == dataset and row["routing_mode"] == "full"
        ])
        for mode_index, mode in enumerate(("full", *study.PRODUCTION_MODES)):
            selected = [
                row for row in participants
                if row["dataset"] == dataset and row["routing_mode"] == mode
            ]
            values = np.asarray([row["uar"] for row in selected])
            entry: dict[str, Any] = {
                "n_participants": len(values),
                "mean_uar": float(values.mean()),
                "sd_uar": float(values.std(ddof=1)),
                "mean_accuracy": float(np.mean([row["accuracy"] for row in selected])),
                "mean_macro_f1": float(np.mean([row["macro_f1"] for row in selected])),
                "mean_brier": float(np.mean([row["brier"] for row in selected])),
                "mean_ece": float(np.mean([row["ece"] for row in selected])),
            }
            if mode != "full":
                delta = full - values
                ci = bootstrap_mean_ci(delta, 9041 + 100 * dataset_index + mode_index)
                try:
                    p_value = float(wilcoxon(delta, alternative="two-sided").pvalue)
                except ValueError:
                    p_value = 1.0
                entry.update({
                    "full_minus_ablation_mean_uar": float(delta.mean()),
                    "full_minus_ablation_median_uar": float(np.median(delta)),
                    "full_minus_ablation_95ci": list(ci),
                    "wilcoxon_two_sided_p": p_value,
                })
            summary[dataset][mode] = entry
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(path: Path, participants: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    labels = {
        "uniform": "Uniform",
        "shared_score": "Shared score",
        "structural_only": "Structural only",
        "corrective_only": "Corrective only",
    }
    colors = {"seed3": "#2864A6", "seed4": "#D36B32"}
    modes = list(study.PRODUCTION_MODES)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    x = np.arange(len(modes))
    width = 0.34
    for dataset_index, dataset in enumerate(("seed3", "seed4")):
        means = [summary[dataset][mode]["mean_uar"] * 100 for mode in modes]
        full_mean = summary[dataset]["full"]["mean_uar"] * 100
        offset = (-0.5 if dataset_index == 0 else 0.5) * width
        axes[0].bar(
            x + offset, means, width=width, color=colors[dataset], alpha=0.84,
            label="SEED" if dataset == "seed3" else "SEED-IV",
        )
        axes[0].axhline(full_mean, color=colors[dataset], linewidth=1.1, linestyle="--")
    axes[0].set_xticks(x, [labels[mode] for mode in modes], rotation=22, ha="right")
    axes[0].set_ylabel("Participant-level UAR (%)")
    axes[0].set_title("a  Component removal")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[0].spines[["top", "right"]].set_visible(False)

    rng = np.random.default_rng(731)
    positions = []
    tick_labels = []
    position = 0
    for mode in modes:
        for dataset in ("seed3", "seed4"):
            full = np.asarray([
                row["uar"] for row in participants
                if row["dataset"] == dataset and row["routing_mode"] == "full"
            ])
            ablated = np.asarray([
                row["uar"] for row in participants
                if row["dataset"] == dataset and row["routing_mode"] == mode
            ])
            delta = (full - ablated) * 100
            jitter = rng.uniform(-0.12, 0.12, len(delta))
            axes[1].scatter(
                np.full(len(delta), position) + jitter, delta, s=12,
                color=colors[dataset], alpha=0.55, linewidths=0,
            )
            median = float(np.median(delta))
            axes[1].plot([position - 0.19, position + 0.19], [median, median],
                         color="black", linewidth=1.4)
            positions.append(position)
            tick_labels.append(("S" if dataset == "seed3" else "S-IV"))
            position += 1
        position += 0.45
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(positions, tick_labels)
    axes[1].set_ylabel("Full minus ablation UAR (points)")
    axes[1].set_title("b  Participant-paired effects")
    axes[1].spines[["top", "right"]].set_visible(False)
    for index, mode in enumerate(modes):
        center = 2 * index + 0.5 + 0.45 * index
        axes[1].text(center, axes[1].get_ylim()[0], labels[mode], rotation=22,
                     ha="right", va="top", fontsize=6.5, clip_on=False)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def make_tex(path: Path, summary: dict[str, Any]) -> None:
    labels = {
        "uniform": "Uniform weighting",
        "shared_score": "Single shared score",
        "structural_only": "Structural route only",
        "corrective_only": "Corrective route only",
        "full": "Full dual-role routing",
    }
    lines = []
    for mode in ("uniform", "shared_score", "structural_only", "corrective_only", "full"):
        values = []
        for dataset in ("seed3", "seed4"):
            entry = summary[dataset][mode]
            values.append(f"${100*entry['mean_uar']:.2f}\\pm{100*entry['sd_uar']:.2f}$")
        lines.append(f"{labels[mode]} & {values[0]} & {values[1]} \\\\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "production"), required=True)
    args = parser.parse_args()
    run_tag = study.SMOKE_TAG if args.phase == "smoke" else study.PRODUCTION_TAG
    modes = ROUTING_MODES_FOR_PHASE = (
        study.ROUTING_MODES if args.phase == "smoke" else study.PRODUCTION_MODES
    )
    folds = expected_folds(args.phase)
    rows: list[dict[str, Any]] = []
    technical: list[dict[str, Any]] = []
    for mode in modes:
        for dataset, session, subject in folds:
            row, audit = audit_ablation_fold(
                run_tag, mode, dataset, session, subject
            )
            rows.append(row)
            technical.append(audit)
    full_control_max = 0.0
    if args.phase == "smoke":
        differences = [
            full_control_difference(run_tag, dataset, session, subject)
            for dataset, session, subject in folds
        ]
        full_control_max = max(differences)
        if full_control_max > 5e-7:
            raise RuntimeError(
                f"full-control parent equivalence failed: {full_control_max}"
            )
        summary = {}
        participants = []
    else:
        rows.extend(load_parent_row(*fold) for fold in folds)
        participants = participant_rows(rows)
        summary = summarize(participants)

    output = HERE / "results" / study.STUDY / run_tag / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "fold_metrics.csv", rows)
    if participants:
        write_csv(output / "participant_metrics.csv", participants)
        make_figure(output / "core_routing_ablation.pdf", participants, summary)
        make_tex(output / "core_routing_ablation_rows.tex", summary)
    audit = {
        "study": study.STUDY,
        "phase": args.phase,
        "run_tag": run_tag,
        "passed": True,
        "expected_ablation_folds": len(folds) * len(modes),
        "observed_ablation_folds": len(technical),
        "imported_parent_folds": len(folds) if args.phase == "production" else 0,
        "verified_ablation_hashes": int(sum(row["verified_hashes"] for row in technical)),
        "max_reconstruction_abs_diff": max(
            row["reconstruction_max_abs_diff"] for row in technical
        ),
        "max_metric_abs_diff": max(row["metric_max_abs_diff"] for row in technical),
        "smoke_full_control_parent_max_abs_diff": full_control_max,
        "summary": summary,
        "code_sha256": study.code_provenance(),
        "git": study.git_provenance(),
    }
    stage_e1.atomic_json(output / "final_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
