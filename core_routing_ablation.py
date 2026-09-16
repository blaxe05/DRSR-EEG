"""Outer-only, label-isolated core routing ablation for sealed DRSR folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

import upstream_v2_js_relevance as stage_e1
import upstream_v2_pseudo_conditional as stage_e2
from data_utils._get_dataset import get_dataset
from models.RWHEDN import RWHEDN
from models.RWHEDNRoutingAblation import ROUTING_MODES, RWHEDNRoutingAblation
from seed_series_upstream_oracle import build_args, validate_dataset
from trainers import HEDNTrainer
from utils.utils import setup_seed


HERE = Path(__file__).resolve().parent
STUDY = "core_routing_ablation"
PROTOCOL_VERSION = "v2"
PARENT_TAG = "e2r12p_20260820"
SMOKE_TAG = "core_routing_ablation_smoke_r2_20260906"
PRODUCTION_TAG = "core_routing_ablation_r2_20260906"
ALLOWED_TAGS = {SMOKE_TAG, PRODUCTION_TAG}
PRODUCTION_MODES = tuple(mode for mode in ROUTING_MODES if mode != "full")
ROOT_FILES = ("parent_reference.json", "outer_result.json", "fold_ledger.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def git_provenance() -> dict[str, str]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        revision = "unavailable"
    return {"revision": revision}


def code_provenance() -> dict[str, Any]:
    fixed = [
        "core_routing_ablation.py",
        "analyze_core_routing_ablation.py",
        "run_core_routing_ablation_queue.py",
        "CORE_ROUTING_ABLATION_CONTRACT.md",
        "models/RWHEDNRoutingAblation.py",
        "upstream_v2_js_relevance.py",
        "upstream_v2_pseudo_conditional.py",
        "models/RWHEDNPseudoConditional.py",
        "models/RWHEDNJS.py",
        "models/RWHEDN.py",
        "models/HEDN.py",
        "trainers/HEDNTrainer.py",
        "data_utils/HEDNLoader.py",
        "data_utils/_get_dataset.py",
        "seed_series_upstream_oracle.py",
        "upstream_v2_baseline_parity.py",
        "upstream_v2_residual.py",
        "config.py",
        "hedn.yaml",
        "utils/utils.py",
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


# Reused fit machinery resolves these symbols from its module at runtime.
# The override gives the ablation its own immutable provenance while allowing
# unrelated manuscript edits during the long production run.
stage_e1.STUDY = STUDY
stage_e1.PROTOCOL_VERSION = PROTOCOL_VERSION
stage_e1.code_provenance = code_provenance
stage_e1.git_provenance = git_provenance


def route_mode_from_fit_dir(fit_dir: Path) -> str:
    mode = fit_dir.parents[3].name
    if mode not in ROUTING_MODES:
        raise RuntimeError(f"cannot derive routing mode from {fit_dir}")
    return mode


def parent_fold(dataset: str, session: int, subject: int) -> Path:
    return (
        HERE / "results" / "upstream_v2_pseudo_conditional" / PARENT_TAG
        / dataset / f"session_{session}" / f"target_subject_{subject:02d}"
    )


def load_parent_reference(dataset: str, session: int, subject: int) -> dict[str, Any]:
    root = parent_fold(dataset, session, subject)
    required = [root / "selection.json", root / "outer_result.json", root / "fold_ledger.json"]
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"incomplete E2 parent fold: {root}")
    selection = json.loads(required[0].read_text(encoding="utf-8"))
    result = json.loads(required[1].read_text(encoding="utf-8"))
    ledger = json.loads(required[2].read_text(encoding="utf-8"))
    identity = (dataset, session, subject)
    if (
        (selection.get("dataset"), selection.get("session"), selection.get("outer_subject")) != identity
        or (result.get("dataset"), result.get("session"), result.get("outer_subject")) != identity
        or selection.get("outer_labels_opened") is not False
        or result.get("target_labels_used_for_training_or_selection") is not False
    ):
        raise RuntimeError(f"E2 parent identity or label boundary failed: {root}")
    hashes = ledger.get("sha256", {})
    checked: dict[str, str] = {}
    for relative in (
        "selection.json",
        "outer_result.json",
        "outer_fit/trajectories.npz",
        "outer_fit/training_log.csv",
        "outer_fit/fit_manifest.json",
        "outer_fit/optimization_receipt.json",
        "outer_fit/pipeline_probabilities.npz",
        "outer_fit/target_labels.npz",
        "outer_fit/evaluation.json",
    ):
        path = root / relative
        actual = sha256_file(path)
        if hashes.get(relative) != actual:
            raise RuntimeError(f"E2 parent ledger mismatch: {path}")
        checked[relative] = actual
    return {
        "parent_root": str(root),
        "parent_tag": PARENT_TAG,
        "dataset": dataset,
        "session": session,
        "subject": subject,
        "selected_js_relevance_weight": float(result["selected_js_relevance_weight"]),
        "selected_conditional_alignment_weight": float(
            result["selected_conditional_alignment_weight"]
        ),
        "selected_source_only_warmup_iters": int(
            result["selected_source_only_warmup_iters"]
        ),
        "selected_top_k_requested": int(result["selected_top_k_requested"]),
        "selected_temperature": float(result["selected_temperature"]),
        "selected_gate": float(result["selected_gate"]),
        "selected_outer_metrics": result["selected_outer_metrics"],
        "verified_parent_sha256": checked,
        "parent_fold_ledger_sha256": sha256_file(required[2]),
    }


def build_ablation_trainer(args: argparse.Namespace, mode: str, conditional: float) -> HEDNTrainer:
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
    model = RWHEDNRoutingAblation(
        routing_mode=mode,
        conditional_alignment_weight=conditional,
        conditional_eps=1e-6,
        js_relevance_weight=float(args.js_relevance_weight),
        js_eps=1e-6,
        **params,
        **RWHEDN.stage_flags("l1"),
        sra_temp=1.0,
        rel_momentum=0.9,
    ).to(args.device)
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
        device=args.device,
    )


def annotate_child_artifacts(fit_dir: Path, mode: str, conditional: float) -> None:
    manifest_path = fit_dir / "fit_manifest.json"
    receipt_path = fit_dir / "optimization_receipt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"] = "rwhedn_routing_ablation"
    manifest["routing_mode"] = mode
    manifest["conditional_alignment_weight"] = conditional
    manifest.setdefault("execution_contract", {}).update({
        "routing_assignment_only_ablation": True,
        "routing_mode": mode,
        "structural_and_corrective_route_definition": {
            "uniform": "uniform structural and corrective weights",
            "shared_score": "structural score distribution shared by both roles",
            "structural_only": "structural weights with uniform corrective weights",
            "corrective_only": "uniform structural weights with corrective weights",
            "full": "exact delegation to the sealed parent implementation",
        }[mode],
        "parent_selection_frozen": True,
    })
    atomic_json(manifest_path, manifest)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["routing_mode"] = mode
    receipt["conditional_alignment_weight"] = conditional
    receipt["fit_manifest_sha256"] = sha256_file(manifest_path)
    atomic_json(receipt_path, receipt)


def child_entry(payload: dict[str, Any], sender: Any) -> None:
    try:
        fit_dir = Path(payload["fit_dir"])
        mode = route_mode_from_fit_dir(fit_dir)
        parent = load_parent_reference(
            str(payload["dataset"]), int(payload["session"]), int(payload["outer_subject"])
        )
        conditional = float(parent["selected_conditional_alignment_weight"])
        stage_e1.get_model_utils = lambda args: build_ablation_trainer(
            args, mode, conditional
        )
        result = stage_e1.train_child(payload)
        annotate_child_artifacts(fit_dir, mode, conditional)
        result["routing_mode"] = mode
        result["fit_manifest_sha256"] = sha256_file(fit_dir / "fit_manifest.json")
        result["receipt_sha256"] = sha256_file(fit_dir / "optimization_receipt.json")
        sender.send({"ok": True, "result": result})
    except BaseException as exc:
        try:
            sender.send({
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
        finally:
            sender.close()
        raise
    else:
        sender.close()


stage_e1.child_entry = child_entry


def fold_directory(run_tag: str, mode: str, dataset: str, session: int, subject: int) -> Path:
    return (
        HERE / "results" / STUDY / run_tag / mode / dataset
        / f"session_{session}" / f"target_subject_{subject:02d}"
    )


def claim_path(run_tag: str, mode: str, dataset: str, session: int, subject: int) -> Path:
    identity = f"{run_tag}|{mode}|{dataset}|{session}|{subject}"
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return HERE / "results" / STUDY / ".claims" / f"{key}.lock"


def verify_fit_metadata(fit_dir: Path, mode: str, conditional: float) -> None:
    manifest = json.loads((fit_dir / "fit_manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads((fit_dir / "evaluation.json").read_text(encoding="utf-8"))
    receipt = json.loads((fit_dir / "optimization_receipt.json").read_text(encoding="utf-8"))
    for document in (manifest, receipt):
        if document.get("routing_mode") != mode:
            raise RuntimeError(f"routing-mode metadata mismatch: {fit_dir}")
        if float(document.get("conditional_alignment_weight", -1.0)) != conditional:
            raise RuntimeError(f"conditional metadata mismatch: {fit_dir}")
    evaluation_launch = evaluation.get("launch_provenance", {})
    evaluation_mode = evaluation.get("routing_mode", evaluation_launch.get("routing_mode"))
    evaluation_conditional = evaluation.get(
        "conditional_alignment_weight",
        evaluation_launch.get("conditional_alignment_weight", -1.0),
    )
    if evaluation_mode != mode:
        raise RuntimeError(f"evaluation routing-mode metadata mismatch: {fit_dir}")
    if float(evaluation_conditional) != conditional:
        raise RuntimeError(f"evaluation conditional metadata mismatch: {fit_dir}")
    if (
        "routing_mode" in evaluation
        and evaluation_launch.get("routing_mode", evaluation["routing_mode"])
        != evaluation["routing_mode"]
    ):
        raise RuntimeError(f"conflicting evaluation routing metadata: {fit_dir}")
    if (
        "conditional_alignment_weight" in evaluation
        and float(
            evaluation_launch.get(
                "conditional_alignment_weight",
                evaluation["conditional_alignment_weight"],
            )
        )
        != float(evaluation["conditional_alignment_weight"])
    ):
        raise RuntimeError(f"conflicting evaluation conditional metadata: {fit_dir}")
    if evaluation.get("selected_probability_storage_dtype") != "float64":
        raise RuntimeError(f"lossless selected-probability evidence missing: {fit_dir}")
    if manifest.get("model") != "rwhedn_routing_ablation":
        raise RuntimeError(f"ablation model metadata mismatch: {fit_dir}")


def validate_complete_fold(
    fold_dir: Path, run_tag: str, mode: str, dataset: str, session: int,
    subject: int, launch: dict[str, Any]
) -> bool:
    if not (fold_dir / "outer_result.json").is_file():
        return False
    actual_dirs = {path.name for path in fold_dir.iterdir() if path.is_dir()}
    actual_files = {path.name for path in fold_dir.iterdir() if path.is_file()}
    if actual_dirs != {"outer_fit"} or actual_files != set(ROOT_FILES):
        raise RuntimeError(f"ablation fold lattice mismatch: {fold_dir}")
    if not stage_e1.validate_fit(
        fold_dir / "outer_fit", role="outer", dataset=dataset, session=session,
        outer_subject=subject, target_subject=subject, run_tag=run_tag,
        launch=launch,
    ):
        return False
    parent = json.loads((fold_dir / "parent_reference.json").read_text(encoding="utf-8"))
    verify_fit_metadata(
        fold_dir / "outer_fit", mode,
        float(parent["selected_conditional_alignment_weight"]),
    )
    ledger = json.loads((fold_dir / "fold_ledger.json").read_text(encoding="utf-8"))
    expected = {
        str(path.relative_to(fold_dir)).replace("\\", "/")
        for path in fold_dir.rglob("*")
        if path.is_file() and path.name != "fold_ledger.json"
    }
    if set(ledger.get("sha256", {})) != expected:
        raise RuntimeError(f"ablation ledger lattice mismatch: {fold_dir}")
    for relative, expected_hash in ledger["sha256"].items():
        if sha256_file(fold_dir / relative) != expected_hash:
            raise RuntimeError(f"ablation ledger hash mismatch: {fold_dir / relative}")
    return True


def run_fold(dataset: str, session: int, subject: int, mode: str, run_tag: str) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if run_tag not in ALLOWED_TAGS or mode not in ROUTING_MODES:
        raise ValueError((run_tag, mode))
    if run_tag == PRODUCTION_TAG and mode == "full":
        raise ValueError("production imports full DRSR from E2 rather than rerunning it")
    if dataset not in ("seed3", "seed4") or session not in (1, 2, 3):
        raise ValueError((dataset, session))
    if subject not in range(1, 16):
        raise ValueError(subject)
    launch = {
        "study": STUDY,
        "protocol_version": PROTOCOL_VERSION,
        "run_tag": run_tag,
        "routing_mode": mode,
        "code_sha256": code_provenance(),
        "git": git_provenance(),
        "captured_at": stage_e1.utc_now(),
        "parent_pid": os.getpid(),
        "argv": list(sys.argv),
    }
    fold_dir = fold_directory(run_tag, mode, dataset, session, subject)
    if validate_complete_fold(
        fold_dir, run_tag, mode, dataset, session, subject, launch
    ):
        print(f"complete immutable routing ablation fold: {fold_dir}", flush=True)
        return fold_dir
    lock = claim_path(run_tag, mode, dataset, session, subject)
    token = stage_e1.acquire_claim(lock, {
        "study": STUDY,
        "run_tag": run_tag,
        "routing_mode": mode,
        "dataset": dataset,
        "session": session,
        "subject": subject,
    })
    try:
        fold_dir.mkdir(parents=True, exist_ok=True)
        allowed = {"outer_fit", *ROOT_FILES}
        foreign = {entry.name for entry in fold_dir.iterdir()} - allowed
        if foreign:
            raise RuntimeError(f"foreign ablation fold artifacts: {sorted(foreign)}")
        parent = load_parent_reference(dataset, session, subject)
        atomic_json(fold_dir / "parent_reference.json", parent)

        setup_seed(42)
        args = build_args(dataset, session, "rwhedn", 42, 1000)
        loaded = get_dataset(args)
        data = np.asarray(loaded["data"])
        labels = np.asarray(loaded["labels"], dtype=np.float32)
        groups = np.asarray(loaded["groups"], dtype=np.int16)
        validate_dataset(dataset, session, data, labels, groups)
        outer_mask = groups[:, 0] == subject
        source_mask = ~outer_mask
        target_labels = np.ascontiguousarray(labels[outer_mask])
        target_data = np.ascontiguousarray(data[outer_mask])
        target_groups = np.ascontiguousarray(groups[outer_mask])
        source_data = np.ascontiguousarray(data[source_mask])
        source_labels = np.ascontiguousarray(labels[source_mask])
        source_groups = np.ascontiguousarray(groups[source_mask])
        original_index = np.flatnonzero(outer_mask).astype(np.int32)
        labels[outer_mask] = 0.0
        del loaded, labels
        if len(np.unique(source_groups[:, 0])) != 14:
            raise RuntimeError("outer source count is not 14")

        vault = stage_e1.TargetLabelVault(target_labels)
        evaluation = stage_e1.run_fit(
            fit_dir=fold_dir / "outer_fit",
            vault=vault,
            role="outer",
            dataset=dataset,
            session=session,
            outer_subject=subject,
            target_subject=subject,
            run_tag=run_tag,
            seed=42,
            source_data=source_data,
            source_labels=source_labels,
            source_groups=source_groups,
            target_data=target_data,
            target_groups=target_groups,
            target_original_index=original_index,
            js_relevance_weight=parent["selected_js_relevance_weight"],
            selected_source_only_warmup_iters=parent["selected_source_only_warmup_iters"],
            selected_top_k=parent["selected_top_k_requested"],
            selected_temperature=parent["selected_temperature"],
            selected_gate=parent["selected_gate"],
            baseline_path=stage_e2.baseline_path_for(
                PARENT_TAG, dataset, session, subject, subject, "outer"
            ),
            launch=launch,
            lossless_selected_probability=True,
        )
        evaluation_path = fold_dir / "outer_fit" / "evaluation.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluation["routing_mode"] = mode
        evaluation["conditional_alignment_weight"] = parent[
            "selected_conditional_alignment_weight"
        ]
        evaluation["parent_tag"] = PARENT_TAG
        atomic_json(evaluation_path, evaluation)
        verify_fit_metadata(
            fold_dir / "outer_fit", mode,
            parent["selected_conditional_alignment_weight"],
        )
        completion = stage_e1.verify_current_provenance(
            launch, context="before routing-ablation outer-result commit"
        )
        result = {
            "study": STUDY,
            "protocol_version": PROTOCOL_VERSION,
            "run_tag": run_tag,
            "routing_mode": mode,
            "dataset": dataset,
            "session": session,
            "outer_subject": subject,
            "seed": 42,
            "parent_reference_sha256": sha256_file(
                fold_dir / "parent_reference.json"
            ),
            "selected_outer_metrics": evaluation["selected_metrics"],
            "parent_selected_outer_metrics": parent["selected_outer_metrics"],
            "target_labels_used_for_training_or_selection": False,
            "target_label_isolated": True,
            "outer_labels_opened_after_probability_commit": True,
            "launch_provenance": launch,
            "completion_provenance": completion,
        }
        atomic_json(fold_dir / "outer_result.json", result)
        ledger = {
            str(path.relative_to(fold_dir)).replace("\\", "/"): sha256_file(path)
            for path in sorted(fold_dir.rglob("*"))
            if path.is_file() and path.name != "fold_ledger.json"
        }
        atomic_json(fold_dir / "fold_ledger.json", {
            "study": STUDY,
            "run_tag": run_tag,
            "routing_mode": mode,
            "dataset": dataset,
            "session": session,
            "outer_subject": subject,
            "sha256": ledger,
            "launch_provenance": launch,
            "completion_provenance": completion,
        })
        if not validate_complete_fold(
            fold_dir, run_tag, mode, dataset, session, subject, launch
        ):
            raise RuntimeError("completed ablation fold failed validation")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return fold_dir
    finally:
        stage_e1.release_claim(lock, token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("seed3", "seed4"), required=True)
    parser.add_argument("--session", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--subject", type=int, choices=range(1, 16), required=True)
    parser.add_argument("--routing-mode", choices=ROUTING_MODES, required=True)
    parser.add_argument("--run-tag", choices=sorted(ALLOWED_TAGS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("the contract fixes seed 42")
    run_fold(args.dataset, args.session, args.subject, args.routing_mode, args.run_tag)


if __name__ == "__main__":
    main()
