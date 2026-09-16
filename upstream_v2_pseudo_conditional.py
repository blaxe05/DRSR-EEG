"""Label-isolated Stage E2 pseudo-class-conditional alignment runner.

One invocation owns one outer fold.  Every Stage E1 selection is immutable;
only the conditional-alignment coefficient varies.  Target labels never cross
the spawned optimizer boundary and open only after the complete frozen-pipeline
probability artifact commits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

import upstream_v2_js_relevance as stage_e1
from models.RWHEDN import RWHEDN
from models.RWHEDNPseudoConditional import RWHEDNPseudoConditional
from trainers import HEDNTrainer


HERE = Path(__file__).resolve().parent
STUDY = "upstream_v2_pseudo_conditional"
PROTOCOL_VERSION = "v1"
CONDITIONAL_WEIGHTS = (0.0, 0.1)
JS_WEIGHTS = (0.0, 1.0)
SMOKE_TAG = "e2r10s_20260820"
PAPER_TAG = "e2r12p_20260820"
ALLOWED_TAGS = {SMOKE_TAG, PAPER_TAG}
PHASE_BY_TAG = {SMOKE_TAG: "smoke", PAPER_TAG: "production"}
E1_TAGS = {
    "smoke": "smoke_upstream_v2_js_relevance_v1",
    "production": "paper_upstream_v2_js_relevance_v1",
}
E1_ALLOWED_TAGS = {
    "smoke": {"smoke_upstream_v2_js_relevance_v1"},
    "production": {
        "paper_upstream_v2_js_relevance_v1",
    },
}
E1_SUMMARY_PATHS = {
    "smoke": HERE / "results" / "upstream_v2_js_relevance" / "analysis" / "smoke" / "summary.json",
    "production": HERE / "results" / "upstream_v2_js_relevance" / "analysis" / "production" / "summary.json",
}
FIT_FILES = stage_e1.FIT_FILES
ROOT_FILES = stage_e1.ROOT_FILES
TOP_K = stage_e1.TOP_K


atomic_json = stage_e1.atomic_json
atomic_npz = stage_e1.atomic_npz
sha256_array = stage_e1.sha256_array
sha256_file = stage_e1.sha256_file
utc_now = stage_e1.utc_now
TargetLabelVault = stage_e1.TargetLabelVault
DATASET_CONTRACT = stage_e1.DATASET_CONTRACT


def git_provenance() -> dict[str, Any]:
    return stage_e1.git_provenance()


def code_provenance() -> dict[str, Any]:
    fixed = [
        "upstream_v2_pseudo_conditional.py",
        "UPSTREAM_V2_PSEUDO_CONDITIONAL_CONTRACT.md",
        "models/RWHEDNPseudoConditional.py",
        "upstream_v2_js_relevance.py",
        "UPSTREAM_V2_JS_RELEVANCE_CONTRACT.md",
        "upstream_v2_residual.py",
        "upstream_v2_baseline_parity.py",
        "seed_series_upstream_oracle.py",
        "config.py",
        "hedn.yaml",
        "get_model_utils.py",
        "models/__init__.py",
        "models/HEDN.py",
        "models/RWHEDN.py",
        "models/RWHEDNJS.py",
        "trainers/HEDNTrainer.py",
        "data_utils/HEDNLoader.py",
        "data_utils/_get_dataset.py",
        "utils/utils.py",
        "datasets/__init__.py",
        "datasets/seed_feature.py",
        "datasets/seediv_feature.py",
    ]
    recursive = sorted(
        str(path.relative_to(HERE)).replace("\\", "/")
        for path in (HERE / "loss_funcs").rglob("*.py")
    )
    files = fixed + recursive
    per_file = {name: sha256_file(HERE / name) for name in files}
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode("utf-8"))
        digest.update(per_file[name].encode("ascii"))
    return {"combined_sha256": digest.hexdigest(), "files": per_file}


def verify_current_provenance(
    launch: dict[str, Any], *, context: str
) -> dict[str, Any]:
    code = code_provenance()
    git = git_provenance()
    if code != launch["code_sha256"]:
        raise RuntimeError(f"code provenance changed {context}")
    if git != launch["git"]:
        raise RuntimeError(f"Git provenance changed {context}")
    return {
        "code_sha256": code,
        "git": git,
        "context": context,
        "checked_at": utc_now(),
    }


# The reused Stage E1 fit machinery resolves these names from its module at
# runtime.  Patching only the in-process module leaves every Stage E1 file
# immutable while giving E2 its own provenance and study identity.
stage_e1.STUDY = STUDY
stage_e1.PROTOCOL_VERSION = PROTOCOL_VERSION
stage_e1.code_provenance = code_provenance


def conditional_directory(weight: float) -> str:
    if weight not in CONDITIONAL_WEIGHTS:
        raise RuntimeError(f"conditional weight outside locked lattice: {weight}")
    return f"cond_{int(round(100.0 * weight)):03d}"


def fold_directory(run_tag: str, dataset: str, session: int, subject: int) -> Path:
    return (
        HERE / "results" / STUDY / run_tag / dataset / f"session_{session}"
        / f"target_subject_{subject:02d}"
    )


def claim_path(run_tag: str, dataset: str, session: int, subject: int) -> Path:
    identity = f"{run_tag}|{dataset}|{session}|{subject}"
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return HERE / "results" / STUDY / ".claims" / f"{key}.lock"


def frozen_pipeline(
    run_tag: str, dataset: str, session: int, subject: int
) -> dict[str, Any]:
    phase = PHASE_BY_TAG[run_tag]
    summary_path = E1_SUMMARY_PATHS[phase]
    if not summary_path.is_file():
        raise RuntimeError(f"Stage E1 analysis missing: {summary_path}")
    valid = (
        summary.get("passed") is True
        or summary.get("completed") is True
        or summary.get("success") is True
        or "selection" in summary
    )
    if not valid or summary.get("phase") != phase:
        raise RuntimeError(f"Stage E1 analysis validation incomplete for E2 {phase}")

    root = (
        HERE / "results" / "upstream_v2_js_relevance" / E1_TAGS[phase]
        / dataset / f"session_{session}" / f"target_subject_{subject:02d}"
    )
    selection_path = root / "selection.json"
    result_path = root / "outer_result.json"
    ledger_path = root / "fold_ledger.json"
    for path in (selection_path, result_path, ledger_path):
        if not path.is_file():
            raise RuntimeError(f"Stage E1 prerequisite missing: {path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    identity = (dataset, session, subject)
    stored_tags = {selection.get("run_tag"), result.get("run_tag")}
    if (
        (selection.get("dataset"), selection.get("session"), selection.get("outer_subject")) != identity
        or (result.get("dataset"), result.get("session"), result.get("outer_subject")) != identity
        or selection.get("outer_labels_opened") is not False
        or len(stored_tags) != 1
        or not stored_tags.issubset(E1_ALLOWED_TAGS[phase])
    ):
        raise RuntimeError(f"Stage E1 identity contract failed: {root}")
    ledger_hashes = ledger.get("sha256", {})
    if (
        ledger_hashes.get("selection.json") != sha256_file(selection_path)
        or ledger_hashes.get("outer_result.json") != sha256_file(result_path)
    ):
        raise RuntimeError(f"Stage E1 root ledger mismatch: {root}")
    selected_js = float(selection["selected_js_relevance_weight"])
    if selected_js not in JS_WEIGHTS or float(result["selected_js_relevance_weight"]) != selected_js:
        raise RuntimeError(f"Stage E1 JS selection mismatch: {root}")
    prior = selection.get("frozen_pipeline", {})
    frozen = {
        "selected_js_relevance_weight": selected_js,
        "selected_top_k_requested": int(prior["selected_top_k_requested"]),
        "selected_temperature": float(prior["selected_temperature"]),
        "selected_gate": float(prior["selected_gate"]),
        "selected_source_only_warmup_iters": int(
            prior["selected_source_only_warmup_iters"]
        ),
        "stage_e1_root": str(root),
        "stage_e1_run_tag": selection["run_tag"],
        "stage_e1_analysis_summary": str(summary_path),
        "sha256": {
            "analysis/summary.json": sha256_file(summary_path),
            "selection.json": sha256_file(selection_path),
            "outer_result.json": sha256_file(result_path),
            "fold_ledger.json": sha256_file(ledger_path),
        },
    }
    if (
        frozen["selected_top_k_requested"] not in TOP_K
        or frozen["selected_temperature"] not in (0.1, 0.25, 0.5, 1.0)
        or frozen["selected_gate"] not in (0.0, 0.25, 0.5, 0.75, 1.0)
        or frozen["selected_source_only_warmup_iters"] not in (0, 50, 100, 200)
    ):
        raise RuntimeError(f"Stage E1 frozen pipeline outside locked lattices: {root}")
    return frozen


def baseline_path_for(
    run_tag: str, dataset: str, session: int, outer: int, target: int, role: str
) -> Path:
    phase = PHASE_BY_TAG[run_tag]
    if role == "inner":
        return (
            HERE / "results" / "upstream_v2_residual_baseline"
            / stage_e1.BASELINE_TAGS[phase] / dataset / f"session_{session}"
            / f"outer_subject_{outer:02d}" / f"inner_target_{target:02d}"
            / "baseline_probabilities.npz"
        )
    return (
        HERE / "results" / "upstream_v2_baseline_parity" / stage_e1.PARITY_TAG
        / dataset / f"session_{session}" / "label_isolated"
        / f"target_subject_{outer:02d}" / "probabilities.npz"
    )


def build_stage_e2_trainer(args: argparse.Namespace, conditional_weight: float) -> HEDNTrainer:
    args.model_name = "rwhedn_pseudo_conditional"
    args.conditional_alignment_weight = float(conditional_weight)
    args.conditional_eps = 1e-6
    params = {
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
    flags = RWHEDN.stage_flags("l1")
    model = RWHEDNPseudoConditional(
        **params,
        **flags,
        sra_temp=getattr(args, "sra_temp", 1.0),
        rel_momentum=getattr(args, "rel_momentum", 0.9),
        js_relevance_weight=getattr(args, "js_relevance_weight", 0.0),
        js_eps=getattr(args, "js_eps", 1e-6),
        conditional_alignment_weight=conditional_weight,
        conditional_eps=1e-6,
    )
    device = getattr(
        args, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device)
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
        early_stop=args.early_stop,
        log_interval=args.log_interval,
        device=device,
    )


def conditional_weight_from_payload(payload: dict[str, Any]) -> float:
    fit_dir = Path(payload["fit_dir"])
    if fit_dir.name.startswith("cond_"):
        weight = int(fit_dir.name.split("_", 1)[1]) / 100.0
    elif fit_dir.name == "outer_fit":
        selection = json.loads(
            (fit_dir.parent / "selection.json").read_text(encoding="utf-8")
        )
        weight = float(selection["selected_conditional_alignment_weight"])
    else:
        raise RuntimeError(f"cannot derive conditional coefficient: {fit_dir}")
    if weight not in CONDITIONAL_WEIGHTS:
        raise RuntimeError(f"derived conditional coefficient outside lattice: {weight}")
    return weight


def child_entry(payload: dict[str, Any], sender: Any) -> None:
    try:
        conditional_weight = conditional_weight_from_payload(payload)
        _install_conditional_npz_writer(conditional_weight)
        _install_conditional_json_writer(conditional_weight)
        stage_e1.get_model_utils = lambda args: build_stage_e2_trainer(
            args, conditional_weight
        )
        result = stage_e1.train_child(payload)
        sender.send({"ok": True, "result": result})
    except BaseException as exc:
        try:
            sender.send(
                {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}
            )
        finally:
            sender.close()
        raise
    else:
        sender.close()


_E2_ATOMIC_NPZ_BASE = stage_e1.atomic_npz
_E2_ATOMIC_JSON_BASE = stage_e1.atomic_json


def _install_conditional_npz_writer(weight: float) -> None:
    def write(path: Path, **arrays: np.ndarray) -> None:
        if Path(path).name in {"trajectories.npz", "pipeline_probabilities.npz"}:
            arrays.setdefault(
                "conditional_alignment_weight",
                np.asarray([weight], dtype=np.float64),
            )
        _E2_ATOMIC_NPZ_BASE(path, **arrays)

    stage_e1.atomic_npz = write


def _install_conditional_json_writer(weight: float) -> None:
    def write(path: Path, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        name = Path(path).name
        if name == "fit_manifest.json":
            payload["study"] = STUDY
            payload["protocol_version"] = PROTOCOL_VERSION
            payload["model"] = "rwhedn_pseudo_conditional"
            payload["conditional_alignment_weight"] = weight
            payload["selected_stage_e1_js_relevance_weight"] = float(
                payload["js_relevance_weight"]
            )
            contract = payload.setdefault("execution_contract", {})
            contract.update(
                {
                    "stage_e2_pseudo_conditional_enabled": bool(weight > 0.0),
                    "stage_e2_zero_delegates_exact_stage_e1": bool(weight == 0.0),
                    "pseudo_class_conditional_alignment_definition": (
                        "per-source source true-class feature mean versus "
                        "unlabeled-target soft-classifier-probability-weighted "
                        "feature mean; class MSE averaged and source-weighted "
                        "by corrective hard weights"
                    ),
                    "confidence_mass_support_gating_enabled": False,
                    "class_specific_source_reliability_enabled": False,
                    "later_stage_e_terms_enabled": False,
                }
            )
        elif name == "optimization_receipt.json":
            payload["study"] = STUDY
            payload["protocol_version"] = PROTOCOL_VERSION
            payload["conditional_alignment_weight"] = weight
        elif name == "evaluation.json":
            payload["study"] = STUDY
            payload["protocol_version"] = PROTOCOL_VERSION
            payload["conditional_alignment_weight"] = weight
            payload["selected_stage_e1_js_relevance_weight"] = float(
                payload["js_relevance_weight"]
            )
        _E2_ATOMIC_JSON_BASE(path, payload)

    stage_e1.atomic_json = write
    global atomic_json
    atomic_json = write


def _rewrite_npz_with_conditional(path: Path, weight: float) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if "conditional_alignment_weight" not in archive.files:
            raise RuntimeError(f"immutable NPZ lacks conditional coefficient: {path}")
        actual = float(archive["conditional_alignment_weight"][0])
    if actual != weight:
        raise RuntimeError(
            f"immutable NPZ conditional coefficient mismatch: {path} expected={weight} actual={actual}"
        )


def materialize_fit_metadata(fit_dir: Path, weight: float) -> dict[str, Any]:
    trajectory = fit_dir / "trajectories.npz"
    pipeline = fit_dir / "pipeline_probabilities.npz"
    manifest_path = fit_dir / "fit_manifest.json"
    receipt_path = fit_dir / "optimization_receipt.json"
    evaluation_path = fit_dir / "evaluation.json"
    _rewrite_npz_with_conditional(trajectory, weight)
    _rewrite_npz_with_conditional(pipeline, weight)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["study"] = STUDY
    manifest["protocol_version"] = PROTOCOL_VERSION
    manifest["model"] = "rwhedn_pseudo_conditional"
    manifest["conditional_alignment_weight"] = weight
    manifest["selected_stage_e1_js_relevance_weight"] = float(
        manifest["js_relevance_weight"]
    )
    contract = manifest.setdefault("execution_contract", {})
    contract.update(
        {
            "stage_e2_pseudo_conditional_enabled": bool(weight > 0.0),
            "stage_e2_zero_delegates_exact_stage_e1": bool(weight == 0.0),
            "pseudo_class_conditional_alignment_definition": (
                "per-source source true-class feature mean versus unlabeled-target "
                "soft-classifier-probability-weighted feature mean; class MSE averaged "
                "and source-weighted by corrective hard weights"
            ),
            "confidence_mass_support_gating_enabled": False,
            "class_specific_source_reliability_enabled": False,
            "later_stage_e_terms_enabled": False,
        }
    )
    atomic_json(manifest_path, manifest)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["study"] = STUDY
    receipt["protocol_version"] = PROTOCOL_VERSION
    receipt["conditional_alignment_weight"] = weight
    receipt["trajectory_sha256"] = sha256_file(trajectory)
    receipt["fit_manifest_sha256"] = sha256_file(manifest_path)
    atomic_json(receipt_path, receipt)

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["study"] = STUDY
    evaluation["protocol_version"] = PROTOCOL_VERSION
    evaluation["conditional_alignment_weight"] = weight
    evaluation["selected_stage_e1_js_relevance_weight"] = float(
        evaluation["js_relevance_weight"]
    )
    evaluation["artifact_sha256"] = {
        "trajectories.npz": sha256_file(trajectory),
        "training_log.csv": sha256_file(fit_dir / "training_log.csv"),
        "fit_manifest.json": sha256_file(manifest_path),
        "optimization_receipt.json": sha256_file(receipt_path),
        "pipeline_probabilities.npz": sha256_file(pipeline),
        "target_labels.npz": sha256_file(fit_dir / "target_labels.npz"),
    }
    atomic_json(evaluation_path, evaluation)
    return evaluation


def validate_fit(
    fit_dir: Path,
    *,
    role: str,
    dataset: str,
    session: int,
    outer_subject: int,
    target_subject: int,
    run_tag: str,
    launch: dict[str, Any],
) -> bool:
    if not fit_dir.exists():
        return False
    actual = {entry.name for entry in fit_dir.iterdir()}
    if not actual:
        return False
    if actual != set(FIT_FILES):
        raise RuntimeError(f"partial/foreign fit artifacts at {fit_dir}: {sorted(actual)}")
    manifest = json.loads((fit_dir / "fit_manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((fit_dir / "optimization_receipt.json").read_text(encoding="utf-8"))
    evaluation = json.loads((fit_dir / "evaluation.json").read_text(encoding="utf-8"))
    expected = (role, dataset, session, outer_subject, target_subject, run_tag)
    for document in (manifest, receipt, evaluation):
        identity = (
            document.get("role"),
            document.get("dataset"),
            document.get("session"),
            document.get("outer_subject"),
            document.get("target_subject"),
            document.get("run_tag"),
        )
        if identity != expected or document.get("study") != STUDY:
            raise RuntimeError(f"fit identity mismatch at {fit_dir}: {identity}")
    weight = float(manifest.get("conditional_alignment_weight", -1.0))
    if weight not in CONDITIONAL_WEIGHTS:
        raise RuntimeError(f"conditional manifest lattice mismatch at {fit_dir}")
    if any(float(doc.get("conditional_alignment_weight", -1.0)) != weight for doc in (receipt, evaluation)):
        raise RuntimeError(f"conditional coefficient mismatch at {fit_dir}")
    if float(manifest.get("js_relevance_weight", -1.0)) not in JS_WEIGHTS:
        raise RuntimeError(f"frozen Stage E1 JS lattice mismatch at {fit_dir}")
    contract = manifest.get("execution_contract", {})
    if (
        manifest.get("model") != "rwhedn_pseudo_conditional"
        or contract.get("confidence_mass_support_gating_enabled") is not False
        or contract.get("class_specific_source_reliability_enabled") is not False
        or contract.get("target_labels_used_for_training_or_selection") is not False
    ):
        raise RuntimeError(f"isolated Stage E2 contract mismatch at {fit_dir}")
    if receipt.get("trajectory_sha256") != sha256_file(fit_dir / "trajectories.npz"):
        raise RuntimeError(f"trajectory hash mismatch at {fit_dir}")
    if receipt.get("training_log_sha256") != sha256_file(fit_dir / "training_log.csv"):
        raise RuntimeError(f"training log hash mismatch at {fit_dir}")
    if receipt.get("fit_manifest_sha256") != sha256_file(fit_dir / "fit_manifest.json"):
        raise RuntimeError(f"manifest hash mismatch at {fit_dir}")
    for name in (
        "trajectories.npz",
        "training_log.csv",
        "fit_manifest.json",
        "optimization_receipt.json",
        "pipeline_probabilities.npz",
        "target_labels.npz",
    ):
        if evaluation["artifact_sha256"].get(name) != sha256_file(fit_dir / name):
            raise RuntimeError(f"evaluation artifact hash mismatch {name}: {fit_dir}")
    for name in ("trajectories.npz", "pipeline_probabilities.npz"):
        with np.load(fit_dir / name, allow_pickle=False) as archive:
            if float(archive["conditional_alignment_weight"][0]) != weight:
                raise RuntimeError(f"NPZ conditional coefficient mismatch {name}: {fit_dir}")
    for document in (manifest, receipt, evaluation):
        if not stage_e1.same_launch_identity(document.get("launch_provenance", {}), launch):
            raise RuntimeError(f"fit launch provenance mismatch at {fit_dir}")
        completion = document.get("completion_provenance", {})
        if completion.get("code_sha256") != launch["code_sha256"] or completion.get("git") != launch["git"]:
            raise RuntimeError(f"fit completion provenance mismatch at {fit_dir}")
    return True


def run_fit(
    *,
    conditional_weight: float,
    frozen_js_weight: float,
    **kwargs: Any,
) -> dict[str, Any]:
    fit_dir = Path(kwargs["fit_dir"])
    validation = dict(
        role=kwargs["role"],
        dataset=kwargs["dataset"],
        session=kwargs["session"],
        outer_subject=kwargs["outer_subject"],
        target_subject=kwargs["target_subject"],
        run_tag=kwargs["run_tag"],
        launch=kwargs["launch"],
    )
    if validate_fit(fit_dir, **validation):
        existing = json.loads((fit_dir / "evaluation.json").read_text(encoding="utf-8"))
        expected = (
            conditional_weight,
            frozen_js_weight,
            kwargs["selected_source_only_warmup_iters"],
            kwargs["selected_top_k"],
            kwargs["selected_temperature"],
            kwargs["selected_gate"],
        )
        actual = (
            existing.get("conditional_alignment_weight"),
            existing.get("js_relevance_weight"),
            existing.get("selected_source_only_warmup_iters"),
            existing.get("selected_top_k_requested"),
            existing.get("selected_temperature"),
            existing.get("selected_gate"),
        )
        if actual != expected:
            raise RuntimeError(f"resumed Stage E2 frozen-pipeline mismatch: {fit_dir}")
        return existing
    if fit_dir.exists() and any(fit_dir.iterdir()):
        raise RuntimeError(f"incomplete or non-empty Stage E2 fit: {fit_dir}")
    stage_e1.child_entry = child_entry
    stage_e1.STUDY = STUDY
    stage_e1.PROTOCOL_VERSION = PROTOCOL_VERSION
    stage_e1.code_provenance = code_provenance
    _install_conditional_npz_writer(conditional_weight)
    _install_conditional_json_writer(conditional_weight)
    stage_e1.run_fit(js_relevance_weight=frozen_js_weight, **kwargs)
    evaluation = materialize_fit_metadata(fit_dir, conditional_weight)
    if not validate_fit(fit_dir, **validation):
        raise RuntimeError(f"materialized Stage E2 fit failed validation: {fit_dir}")
    return evaluation


def lower_tail(values: list[float]) -> tuple[float, float]:
    return stage_e1.lower_tail(values)


def select_conditional(inner_evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for weight in CONDITIONAL_WEIGHTS:
        metrics = [
            row["selected_metrics"]
            for row in inner_evaluations
            if float(row["conditional_alignment_weight"]) == weight
        ]
        if len(metrics) != 14:
            raise RuntimeError(f"conditional inner lattice incomplete for {weight}: {len(metrics)}/14")
        uars = [float(value["uar"]) for value in metrics]
        cvar, p10 = lower_tail(uars)
        rows.append(
            {
                "conditional_alignment_weight": weight,
                "mean_uar": float(np.mean(uars)),
                "mean_accuracy": float(np.mean([value["accuracy"] for value in metrics])),
                "cvar20_uar": cvar,
                "p10_uar": p10,
                "mean_ece": float(
                    np.mean([value["ece_equal_mass_10"] for value in metrics])
                ),
            }
        )
    reference = next(row for row in rows if row["conditional_alignment_weight"] == 0.0)
    tolerance = 1e-10
    for row in rows:
        delta = {
            "accuracy": row["mean_accuracy"] - reference["mean_accuracy"],
            "cvar20_uar": row["cvar20_uar"] - reference["cvar20_uar"],
            "p10_uar": row["p10_uar"] - reference["p10_uar"],
            "ece": row["mean_ece"] - reference["mean_ece"],
        }
        row["guardrail_deltas_vs_zero"] = delta
        row["eligible"] = (
            delta["accuracy"] >= -tolerance
            and delta["cvar20_uar"] >= -tolerance
            and delta["p10_uar"] >= -tolerance
            and delta["ece"] <= tolerance
        )
    winner = max(
        (row for row in rows if row["eligible"]),
        key=lambda row: (
            row["mean_uar"],
            row["mean_accuracy"],
            row["cvar20_uar"],
            row["p10_uar"],
            -row["mean_ece"],
            -row["conditional_alignment_weight"],
        ),
    )
    return {
        "selection_metric": "mean_inner_pseudotarget_uar_with_zero_conditional_guardrails",
        "guardrail_reference_conditional_alignment_weight": 0.0,
        "guardrail_tolerance": tolerance,
        "tie_break": [
            "mean_uar",
            "mean_accuracy",
            "cvar20_uar",
            "p10_uar",
            "lowest_mean_ece",
            "lower_conditional_weight",
        ],
        "candidate_rows": rows,
        "selected_conditional_alignment_weight": float(
            winner["conditional_alignment_weight"]
        ),
    }


def validate_complete_outer_fold(
    fold_dir: Path,
    *,
    run_tag: str,
    dataset: str,
    session: int,
    subject: int,
    launch: dict[str, Any],
) -> bool:
    if not fold_dir.exists() or not (fold_dir / "outer_result.json").exists():
        return False
    expected_dirs = {
        f"inner_target_{value:02d}" for value in range(1, 16) if value != subject
    }
    expected_dirs.add("outer_fit")
    if (
        {entry.name for entry in fold_dir.iterdir() if entry.is_dir()} != expected_dirs
        or {entry.name for entry in fold_dir.iterdir() if entry.is_file()} != set(ROOT_FILES)
    ):
        raise RuntimeError(f"Stage E2 outer-fold artifact lattice mismatch: {fold_dir}")
    for pseudo in range(1, 16):
        if pseudo == subject:
            continue
        inner_dir = fold_dir / f"inner_target_{pseudo:02d}"
        if {path.name for path in inner_dir.iterdir()} != {
            conditional_directory(value) for value in CONDITIONAL_WEIGHTS
        }:
            raise RuntimeError(f"Stage E2 inner lattice mismatch: {inner_dir}")
        for weight in CONDITIONAL_WEIGHTS:
            validate_fit(
                inner_dir / conditional_directory(weight),
                role="inner",
                dataset=dataset,
                session=session,
                outer_subject=subject,
                target_subject=pseudo,
                run_tag=run_tag,
                launch=launch,
            )
    validate_fit(
        fold_dir / "outer_fit",
        role="outer",
        dataset=dataset,
        session=session,
        outer_subject=subject,
        target_subject=subject,
        run_tag=run_tag,
        launch=launch,
    )
    result = json.loads((fold_dir / "outer_result.json").read_text(encoding="utf-8"))
    if (
        result.get("run_tag"),
        result.get("dataset"),
        result.get("session"),
        result.get("outer_subject"),
    ) != (run_tag, dataset, session, subject):
        raise RuntimeError(f"Stage E2 outer-result identity mismatch: {fold_dir}")
    return True


def run_outer_fold(dataset: str, session: int, subject: int, run_tag: str, seed: int) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if run_tag not in ALLOWED_TAGS or dataset not in DATASET_CONTRACT:
        raise ValueError((run_tag, dataset))
    if session not in (1, 2, 3) or subject not in range(1, 16) or seed != 42:
        raise ValueError((session, subject, seed))
    launch = {
        "code_sha256": code_provenance(),
        "git": git_provenance(),
        "captured_at": utc_now(),
        "parent_pid": os.getpid(),
        "argv": list(sys.argv),
    }
    fold_dir = fold_directory(run_tag, dataset, session, subject)
    if validate_complete_outer_fold(
        fold_dir,
        run_tag=run_tag,
        dataset=dataset,
        session=session,
        subject=subject,
        launch=launch,
    ):
        print(f"complete immutable Stage E2 outer fold: {fold_dir}", flush=True)
        return fold_dir
    lock = claim_path(run_tag, dataset, session, subject)
    token = stage_e1.acquire_claim(
        lock,
        {
            "study": STUDY,
            "run_tag": run_tag,
            "dataset": dataset,
            "session": session,
            "outer_subject": subject,
        },
    )
    try:
        fold_dir.mkdir(parents=True, exist_ok=True)
        allowed = {
            f"inner_target_{value:02d}" for value in range(1, 16) if value != subject
        } | {"outer_fit", *ROOT_FILES}
        foreign = {entry.name for entry in fold_dir.iterdir()} - allowed
        if foreign:
            raise RuntimeError(f"foreign Stage E2 outer-fold artifacts: {sorted(foreign)}")

        stage_e1.setup_seed(seed)
        args = stage_e1.build_args(dataset, session, "rwhedn", seed, 1000)
        loaded = stage_e1.get_dataset(args)
        data = np.asarray(loaded["data"])
        labels = np.asarray(loaded["labels"], dtype=np.float32)
        groups = np.asarray(loaded["groups"], dtype=np.int16)
        stage_e1.validate_dataset(dataset, session, data, labels, groups)
        subject_id = groups[:, 0]
        outer_mask = subject_id == subject
        source_mask = ~outer_mask
        outer_labels_hidden = np.ascontiguousarray(labels[outer_mask])
        outer_data = np.ascontiguousarray(data[outer_mask])
        outer_groups = np.ascontiguousarray(groups[outer_mask])
        outer_original = np.flatnonzero(outer_mask).astype(np.int32)
        source_data_all = np.ascontiguousarray(data[source_mask])
        source_labels_all = np.ascontiguousarray(labels[source_mask])
        source_groups_all = np.ascontiguousarray(groups[source_mask])
        labels[outer_mask] = 0.0
        del loaded, labels

        outer_sources = sorted(np.unique(source_groups_all[:, 0]).astype(int).tolist())
        if len(outer_sources) != 14 or subject in outer_sources:
            raise RuntimeError("Stage E2 outer source lattice mismatch")
        frozen = frozen_pipeline(run_tag, dataset, session, subject)
        frozen_js = float(frozen["selected_js_relevance_weight"])
        frozen_top_k = int(frozen["selected_top_k_requested"])
        frozen_temperature = float(frozen["selected_temperature"])
        frozen_gate = float(frozen["selected_gate"])
        frozen_warmup = int(frozen["selected_source_only_warmup_iters"])

        inner_evaluations: list[dict[str, Any]] = []
        for pseudo in outer_sources:
            inner_target_mask = source_groups_all[:, 0] == pseudo
            inner_source_mask = ~inner_target_mask
            inner_dir = fold_dir / f"inner_target_{pseudo:02d}"
            inner_dir.mkdir(parents=True, exist_ok=True)
            for weight in CONDITIONAL_WEIGHTS:
                evaluation = run_fit(
                    conditional_weight=weight,
                    frozen_js_weight=frozen_js,
                    fit_dir=inner_dir / conditional_directory(weight),
                    vault=TargetLabelVault(source_labels_all[inner_target_mask]),
                    role="inner",
                    dataset=dataset,
                    session=session,
                    outer_subject=subject,
                    target_subject=pseudo,
                    run_tag=run_tag,
                    seed=seed,
                    source_data=source_data_all[inner_source_mask],
                    source_labels=source_labels_all[inner_source_mask],
                    source_groups=source_groups_all[inner_source_mask],
                    target_data=source_data_all[inner_target_mask],
                    target_groups=source_groups_all[inner_target_mask],
                    target_original_index=np.flatnonzero(source_mask)[inner_target_mask].astype(np.int32),
                    selected_source_only_warmup_iters=frozen_warmup,
                    selected_top_k=frozen_top_k,
                    selected_temperature=frozen_temperature,
                    selected_gate=frozen_gate,
                    baseline_path=baseline_path_for(
                        run_tag, dataset, session, subject, pseudo, "inner"
                    ),
                    launch=launch,
                )
                inner_evaluations.append(evaluation)
                print(
                    f"[{dataset}/s{session}/outer{subject:02d}] inner {pseudo:02d} "
                    f"conditional_weight={weight:g} complete",
                    flush=True,
                )

        selection = {
            "study": STUDY,
            "protocol_version": PROTOCOL_VERSION,
            "run_tag": run_tag,
            "dataset": dataset,
            "session": session,
            "outer_subject": subject,
            "inner_pseudo_target_subjects": outer_sources,
            "outer_labels_opened": False,
            "conditional_alignment_candidates": list(CONDITIONAL_WEIGHTS),
            "fixed_final_iteration": 1000,
            "frozen_pipeline": frozen,
            **select_conditional(inner_evaluations),
            "launch_provenance": launch,
        }
        atomic_json(fold_dir / "selection.json", selection)
        selected_weight = float(selection["selected_conditional_alignment_weight"])
        outer_evaluation = run_fit(
            conditional_weight=selected_weight,
            frozen_js_weight=frozen_js,
            fit_dir=fold_dir / "outer_fit",
            vault=TargetLabelVault(outer_labels_hidden),
            role="outer",
            dataset=dataset,
            session=session,
            outer_subject=subject,
            target_subject=subject,
            run_tag=run_tag,
            seed=seed,
            source_data=source_data_all,
            source_labels=source_labels_all,
            source_groups=source_groups_all,
            target_data=outer_data,
            target_groups=outer_groups,
            target_original_index=outer_original,
            selected_source_only_warmup_iters=frozen_warmup,
            selected_top_k=frozen_top_k,
            selected_temperature=frozen_temperature,
            selected_gate=frozen_gate,
            baseline_path=baseline_path_for(
                run_tag, dataset, session, subject, subject, "outer"
            ),
            launch=launch,
        )
        completion = verify_current_provenance(
            launch, context="before Stage E2 outer-result commit"
        )
        result = {
            "study": STUDY,
            "protocol_version": PROTOCOL_VERSION,
            "run_tag": run_tag,
            "dataset": dataset,
            "session": session,
            "outer_subject": subject,
            "outer_source_subjects": outer_sources,
            "inner_pseudo_target_subjects": outer_sources,
            "selected_conditional_alignment_weight": selected_weight,
            "selected_js_relevance_weight": frozen_js,
            "selected_source_only_warmup_iters": frozen_warmup,
            "selected_top_k_requested": frozen_top_k,
            "selected_temperature": frozen_temperature,
            "selected_gate": frozen_gate,
            "selection": selection,
            "selected_outer_metrics": outer_evaluation["selected_metrics"],
            "outer_baseline_metrics": outer_evaluation["baseline_metrics"],
            "outer_route_metrics": outer_evaluation["route_metrics"],
            "target_labels_used_for_training_or_selection": False,
            "target_label_isolated": True,
            "outer_labels_opened_after_selection_and_probability_commit": True,
            "launch_provenance": launch,
            "completion_provenance": completion,
        }
        atomic_json(fold_dir / "outer_result.json", result)
        ledger = {}
        for path in sorted(fold_dir.rglob("*")):
            if path.is_file() and path.name != "fold_ledger.json":
                ledger[str(path.relative_to(fold_dir)).replace("\\", "/")] = sha256_file(path)
        atomic_json(
            fold_dir / "fold_ledger.json",
            {
                "study": STUDY,
                "run_tag": run_tag,
                "dataset": dataset,
                "session": session,
                "outer_subject": subject,
                "sha256": ledger,
                "launch_provenance": launch,
                "completion_provenance": completion,
            },
        )
        if not validate_complete_outer_fold(
            fold_dir,
            run_tag=run_tag,
            dataset=dataset,
            session=session,
            subject=subject,
            launch=launch,
        ):
            raise RuntimeError("completed Stage E2 outer fold failed final validation")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return fold_dir
    finally:
        stage_e1.release_claim(lock, token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_CONTRACT), required=True)
    parser.add_argument("--session", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--subject", type=int, choices=range(1, 16), required=True)
    parser.add_argument("--run-tag", choices=sorted(ALLOWED_TAGS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_outer_fold(args.dataset, args.session, args.subject, args.run_tag, args.seed)


if __name__ == "__main__":
    main()
