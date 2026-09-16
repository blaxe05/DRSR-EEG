"""Label-isolated nested normalized-JS relevance runner.

One invocation owns one outer fold. Sparse top-k, temperature, residual gate,
and source-only warm-up choices are immutable inputs; only the normalized-JS
relevance coefficient varies.
Target labels never cross the spawned optimization boundary and are opened
only after the complete frozen-pipeline probability artifact is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import subprocess
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from data_utils._get_dataset import get_dataset
from data_utils.HEDNLoader import HEDNLoader
from get_model_utils import get_model_utils
from seed_series_upstream_oracle import (
    DATASET_CONTRACT,
    UPSTREAM_COMMIT,
    build_args,
    validate_dataset,
    within_trial_index,
)
from upstream_v2_baseline_parity import (
    TargetLabelVault,
    atomic_json,
    atomic_npz,
    atomic_training_log,
    multiclass_metrics,
    sha256_array,
    sha256_file,
    sha256_model_state,
    utc_now,
)
from upstream_v2_residual import (
    assert_identity,
    baseline_probability,
    normalize,
    route_probability,
)
from utils.utils import setup_seed


HERE = Path(__file__).resolve().parent
STUDY = "upstream_v2_js_relevance"
PROTOCOL_VERSION = "v1"
TOP_K = (1, 2, 4, 14)
JS_WEIGHTS = (0.0, 1.0)
CHECKPOINTS = tuple(range(50, 1001, 50))
SMOKE_TAG = "smoke_upstream_v2_js_relevance_v1"
PAPER_TAG = "paper_upstream_v2_js_relevance_v1"
ALLOWED_TAGS = {SMOKE_TAG, PAPER_TAG}
PHASE_BY_TAG = {SMOKE_TAG: "smoke", PAPER_TAG: "production"}
WARMUP_TAGS = {
    "smoke": "smoke_upstream_v2_warmup_v1",
    "production": "paper_upstream_v2_warmup_v1",
}
BASELINE_TAGS = {
    "smoke": "smoke_upstream_v2_residual_baseline_v1",
    "production": "paper_upstream_v2_residual_baseline_v1",
}
PARITY_TAG = "paper_upstream_v2_baseline_parity_label_isolated_v1"
FIT_FILES = (
    "trajectories.npz",
    "training_log.csv",
    "fit_manifest.json",
    "optimization_receipt.json",
    "pipeline_probabilities.npz",
    "target_labels.npz",
    "evaluation.json",
)
ROOT_FILES = ("selection.json", "outer_result.json", "fold_ledger.json")


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                args, cwd=HERE, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return "unavailable"

    status = run("git", "status", "--short")
    return {
        "revision": run("git", "rev-parse", "HEAD"),
        "dirty": bool(status and status != "unavailable"),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def code_provenance() -> dict[str, Any]:
    fixed = [
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


def json_safe_namespace(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        elif isinstance(value, dict):
            result[key] = value
        else:
            result[key] = str(value)
    return result


def verify_current_provenance(
    launch: dict[str, Any], *, context: str
) -> dict[str, Any]:
    code = code_provenance()
    git = git_provenance()
    if code != launch["code_sha256"]:
        raise RuntimeError(f"code provenance changed {context}")
    if git != launch["git"]:
        raise RuntimeError(f"Git provenance changed {context}")
    return {"code_sha256": code, "git": git, "context": context, "checked_at": utc_now()}


def same_launch_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare immutable provenance while allowing a new parent PID/timestamp on resume."""
    return (
        left.get("code_sha256") == right.get("code_sha256")
        and left.get("git") == right.get("git")
    )


def fold_directory(run_tag: str, dataset: str, session: int, subject: int) -> Path:
    return (
        HERE / "results" / STUDY / run_tag / dataset / f"session_{session}" /
        f"target_subject_{subject:02d}"
    )


def claim_path(run_tag: str, dataset: str, session: int, subject: int) -> Path:
    identity = f"{run_tag}|{dataset}|{session}|{subject}"
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return HERE / "results" / STUDY / ".claims" / f"{key}.lock"


def frozen_pipeline(run_tag: str, dataset: str, session: int, subject: int) -> dict[str, Any]:
    """Load the already-selected Stage B/C/D choices without re-selection."""
    phase = PHASE_BY_TAG[run_tag]
    root = (
        HERE / "results" / "upstream_v2_warmup" / WARMUP_TAGS[phase] /
        dataset / f"session_{session}" / f"target_subject_{subject:02d}"
    )
    selection_path = root / "selection.json"
    result_path = root / "outer_result.json"
    ledger_path = root / "fold_ledger.json"
    documents = [selection_path, result_path, ledger_path]
    if not all(path.is_file() for path in documents):
        raise RuntimeError(f"validated warm-up prerequisite missing: {root}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    identity = (selection.get("dataset"), selection.get("session"), selection.get("outer_subject"))
    if identity != (dataset, session, subject) or selection.get("outer_labels_opened") is not False:
        raise RuntimeError("warm-up prerequisite identity/label-vault mismatch")
    prior_pipeline = selection.get("frozen_pipeline", {})
    top_k = int(prior_pipeline["selected_top_k_requested"])
    temperature = float(prior_pipeline["selected_temperature"])
    gate = float(prior_pipeline["selected_gate"])
    warmup = int(selection["selected_source_only_warmup_iters"])
    if (top_k not in TOP_K or temperature not in (0.10, 0.25, 0.50, 1.00)
            or gate not in (0.0, 0.25, 0.5, 0.75, 1.0)
            or warmup not in (0, 50, 100, 200)):
        raise RuntimeError("warm-up prerequisite selected-value lattice mismatch")
    return {
        "selected_top_k_requested": top_k,
        "selected_temperature": temperature,
        "selected_gate": gate,
        "selected_source_only_warmup_iters": warmup,
        "warmup_root": str(root),
        "sha256": {path.name: sha256_file(path) for path in documents},
    }


def js_directory(weight: float) -> str:
    if weight not in JS_WEIGHTS:
        raise RuntimeError(f"JS weight outside locked lattice: {weight}")
    return f"js_{int(round(100.0 * weight)):03d}"


def baseline_path_for(
    run_tag: str, dataset: str, session: int, outer_subject: int, target_subject: int,
    role: str,
) -> Path:
    phase = PHASE_BY_TAG[run_tag]
    if role == "inner":
        return (
            HERE / "results" / "upstream_v2_residual_baseline" / BASELINE_TAGS[phase] /
            dataset / f"session_{session}" / f"outer_subject_{outer_subject:02d}" /
            f"inner_target_{target_subject:02d}" / "baseline_probabilities.npz"
        )
    return (
        HERE / "results" / "upstream_v2_baseline_parity" / PARITY_TAG / dataset /
        f"session_{session}" / "label_isolated" / f"target_subject_{outer_subject:02d}" /
        "probabilities.npz"
    )


def acquire_claim(path: Path, identity: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    raw = (json.dumps({**identity, "token": token, "pid": os.getpid(), "at": utc_now()},
                      sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"outer-fold claim already exists: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return token


def release_claim(path: Path, token: str) -> None:
    if not path.exists():
        raise RuntimeError(f"outer-fold claim not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("token") != token:
        raise RuntimeError(f"outer-fold claim token mismatch: {path}")
    path.unlink()


def source_specific_batched(model: Any, data: np.ndarray, device: Any,
                            batch_size: int = 2048) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(data, dtype=torch.float32)
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            value = model.source_specific_predict_proba(
                tensor[start:start + batch_size].to(device)
            )
            pieces.append(value.detach().cpu().numpy().astype(np.float32, copy=False))
    probabilities = np.concatenate(pieces, axis=1)
    return probabilities / np.clip(probabilities.sum(axis=2, keepdims=True), 1e-12, None)


def dense_batched(model: Any, data: Any, device: Any, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data, dtype=torch.float32)
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            value = model.predict_proba(
                tensor[start:start + batch_size].to(device), mode="target"
            )
            pieces.append(value.detach().cpu().numpy().astype(np.float32, copy=False))
    probabilities = np.concatenate(pieces, axis=0)
    return probabilities / np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)


def retained_source_indices(weights: np.ndarray, source_subjects: list[int], k: int) -> np.ndarray:
    effective = min(int(k), len(source_subjects))
    # Primary key is descending weight; immutable ascending subject id resolves ties.
    order = np.lexsort((np.asarray(source_subjects, dtype=int), -np.asarray(weights, dtype=float)))
    return order[:effective].astype(np.int16)


def candidate_mixtures(
    per_source: np.ndarray, weights: np.ndarray, source_subjects: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mixtures: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    effective: list[int] = []
    for requested in TOP_K:
        selected = retained_source_indices(weights, source_subjects, requested)
        masked = np.zeros(len(source_subjects), dtype=np.float64)
        masked[selected] = np.asarray(weights, dtype=np.float64)[selected]
        if masked.sum() <= 0:
            masked[selected] = 1.0
        masked /= masked.sum()
        mixture = np.einsum("s,snc->nc", masked, per_source, optimize=True)
        mixture /= np.clip(mixture.sum(axis=1, keepdims=True), 1e-12, None)
        mixtures.append(mixture.astype(np.float32))
        masks.append(masked.astype(np.float32))
        effective.append(len(selected))
    return np.stack(mixtures), np.stack(masks), np.asarray(effective, dtype=np.int16)


def train_child(payload: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "fit_dir", "run_tag", "dataset", "session", "outer_subject", "target_subject",
        "role", "seed", "max_iter", "source_data", "source_labels", "source_groups",
        "target_data", "target_groups", "target_original_index", "selected_top_k",
        "js_relevance_weight", "selected_source_only_warmup_iters",
        "input_provenance", "launch_provenance", "launch_argv", "parent_pid",
        "label_vault_state", "label_vault_transitions",
    }
    if set(payload) != expected:
        raise RuntimeError(
            f"optimizer payload mismatch missing={sorted(expected-set(payload))} "
            f"foreign={sorted(set(payload)-expected)}"
        )
    if payload["label_vault_state"] != "sealed":
        raise RuntimeError("optimizer child requires a sealed target-label vault")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    fit_dir = Path(payload["fit_dir"])
    if any(fit_dir.iterdir()):
        raise RuntimeError(f"optimizer child requires empty fit directory: {fit_dir}")
    launch = payload["launch_provenance"]
    verify_current_provenance(launch, context="before sparse optimizer construction")

    dataset = str(payload["dataset"])
    session = int(payload["session"])
    role = str(payload["role"])
    target_subject = int(payload["target_subject"])
    seed = int(payload["seed"])
    max_iter = int(payload["max_iter"])
    js_weight = float(payload["js_relevance_weight"])
    if js_weight not in JS_WEIGHTS:
        raise RuntimeError("normalized-JS weight is outside the locked lattice")
    selected_warmup = int(payload["selected_source_only_warmup_iters"])
    if selected_warmup not in (0, 50, 100, 200):
        raise RuntimeError("frozen source-only warm-up is outside its locked lattice")
    source_data = np.ascontiguousarray(payload["source_data"])
    source_labels = np.ascontiguousarray(payload["source_labels"], dtype=np.float32)
    source_groups = np.ascontiguousarray(payload["source_groups"], dtype=np.int16)
    target_data = np.ascontiguousarray(payload["target_data"])
    target_groups = np.ascontiguousarray(payload["target_groups"], dtype=np.int16)
    source_subjects = sorted(np.unique(source_groups[:, 0]).astype(int).tolist())
    expected_sources = 13 if role == "inner" else 14
    if len(source_subjects) != expected_sources or target_subject in source_subjects:
        raise RuntimeError("source identity/count contract failed")

    setup_seed(seed)
    args = build_args(dataset, session, "rwhedn", seed, max_iter)
    args.model_name = "rwhedn_js"
    args.rw_stage = "l1"
    args.js_relevance_weight = js_weight
    args.js_eps = 1e-6
    args.sra_temp = 1.0
    args.rel_momentum = 0.9
    args.lr = 1e-3
    args.weight_decay = 1e-5
    args.lr_scheduler = False
    args.early_stop = 0
    target_zeros = np.zeros(
        (len(target_data), DATASET_CONTRACT[dataset]["num_classes"]), dtype=np.float32
    )
    setup_seed(seed)
    loader_builder = HEDNLoader(
        args,
        {"data": source_data, "labels": source_labels, "groups": source_groups},
        {"data": target_data, "labels": target_zeros, "groups": target_groups},
    )
    source_loader, target_loader = loader_builder()
    if np.any(target_loader.dataset.d2.numpy()):
        raise RuntimeError("target loader contains nonzero labels")
    retained_mask = DBSCAN(
        eps=float(loader_builder.best_cluster_params["eps"]),
        min_samples=int(loader_builder.best_cluster_params["min_samples"]),
    ).fit(target_data).labels_ != -1
    if int(retained_mask.sum()) != len(target_loader.dataset):
        raise RuntimeError("target retained-mask mismatch")

    trainer = get_model_utils(args)
    initial_state = sha256_model_state(trainer.get_model_state())
    trainer.pre_training_processing(source_loader)
    initialized_state = sha256_model_state(trainer.get_model_state())
    source_iter, target_iter = iter(source_loader), iter(target_loader)
    retained_features = target_loader.dataset.d1
    checkpoint_set = set(CHECKPOINTS)
    per_source_trajectory: list[np.ndarray] = []
    dense_trajectory: list[np.ndarray] = []
    relevance_trajectory: list[np.ndarray] = []
    logs: list[list[float]] = []
    source_resets = 0
    target_resets = 0
    max_dense_reconstruction_diff = 0.0
    started = time.time()

    for iteration in range(1, max_iter + 1):
        trainer.model.train()
        trainer.model.source_only_warmup_active = iteration <= selected_warmup
        try:
            src_data, src_label, src_cluster = next(source_iter)
        except StopIteration:
            source_resets += 1
            source_iter = iter(source_loader)
            src_data, src_label, _ = next(source_iter)  # audited stale-cluster behavior
        try:
            tgt_data, _, tgt_cluster = next(target_iter)
        except StopIteration:
            target_resets += 1
            target_iter = iter(target_loader)
            tgt_data, _, tgt_cluster = next(target_iter)

        src_data = src_data.to(trainer.device)
        src_label = src_label.to(trainer.device)
        tgt_data = tgt_data.to(trainer.device)
        values = trainer.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
        cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss, easy_idx, hard_idx = values
        loss = cls_loss + trainer.transfer_loss_weight * transfer_loss + trainer.constraint_loss_weight * cons_loss
        trainer.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        trainer.optimizer.step()

        trainer.model.source_only_warmup_active = iteration <= selected_warmup
        values = trainer.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
        cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss, easy_idx, hard_idx = values
        trainer.fe_opt.zero_grad()
        (src_clu_loss + tgt_clu_loss).backward()
        trainer.fe_opt.step()

        source_acc = trainer.test(source_loader, mode="source")
        # Preserve the exact per-iteration target evaluation-mode transition,
        # but no labels are available and no checkpoint is selected from it.
        dense_batched(trainer.model, retained_features, trainer.device)
        losses = [cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss]
        logs.append([
            *[float(v.detach().item() if isinstance(v, torch.Tensor) else v) for v in losses],
            float(source_acc), float("nan"), float("nan"),
            float(int(easy_idx.detach().cpu().item()) + 1),
            float(int(hard_idx.detach().cpu().item()) + 1),
        ])

        if iteration in checkpoint_set:
            per_source = source_specific_batched(trainer.model, target_data, trainer.device)
            weights = trainer.model.src_reliability.detach().cpu().numpy().astype(np.float32)
            weights /= np.clip(weights.sum(), 1e-12, None)
            dense = dense_batched(trainer.model, target_data, trainer.device)
            reconstructed = np.einsum("s,snc->nc", weights, per_source, optimize=True)
            reconstructed /= np.clip(reconstructed.sum(axis=1, keepdims=True), 1e-12, None)
            max_dense_reconstruction_diff = max(
                max_dense_reconstruction_diff,
                float(np.max(np.abs(reconstructed - dense))),
            )
            per_source_trajectory.append(per_source)
            relevance_trajectory.append(weights)
            dense_trajectory.append(dense)

    if len(per_source_trajectory) != len(CHECKPOINTS):
        raise RuntimeError("checkpoint trajectory is incomplete")
    per_source_array = np.stack(per_source_trajectory).astype(np.float32)
    relevance_array = np.stack(relevance_trajectory).astype(np.float32)
    dense_array = np.stack(dense_trajectory).astype(np.float32)
    candidates, candidate_weights, effective_top_k = candidate_mixtures(
        per_source_array[-1], relevance_array[-1], source_subjects
    )
    selected_top_k = int(payload["selected_top_k"])
    selected_index = TOP_K.index(selected_top_k) if selected_top_k in TOP_K else -1

    trials = target_groups[:, 1].astype(np.int16)
    original_index = np.asarray(payload["target_original_index"], dtype=np.int32)
    trajectory_path = fit_dir / "trajectories.npz"
    log_path = fit_dir / "training_log.csv"
    manifest_path = fit_dir / "fit_manifest.json"
    receipt_path = fit_dir / "optimization_receipt.json"
    atomic_npz(
        trajectory_path,
        checkpoint_iteration=np.asarray(CHECKPOINTS, dtype=np.int32),
        per_source_probability_trajectory=per_source_array,
        dense_probability_trajectory=dense_array,
        relevance_trajectory=relevance_array,
        candidate_top_k_requested=np.asarray(TOP_K, dtype=np.int16),
        candidate_top_k_effective=effective_top_k,
        candidate_probability_final=candidates,
        candidate_weight_final=candidate_weights,
        selected_top_k_requested=np.asarray([selected_top_k], dtype=np.int16),
        js_relevance_weight=np.asarray([js_weight], dtype=np.float64),
        selected_source_only_warmup_iters=np.asarray([selected_warmup], dtype=np.int32),
        selected_probability_final=(
            candidates[selected_index] if selected_index >= 0
            else np.empty((0, DATASET_CONTRACT[dataset]["num_classes"]), dtype=np.float32)
        ),
        source_subject=np.asarray(source_subjects, dtype=np.int16),
        target_subject=np.full(len(target_data), target_subject, dtype=np.int16),
        target_session=np.full(len(target_data), session, dtype=np.int8),
        target_trial=trials,
        target_sample_within_trial=within_trial_index(trials),
        target_original_dataset_index=original_index,
        retained_mask=retained_mask.astype(np.bool_),
    )
    atomic_training_log(log_path, np.asarray(logs, dtype=np.float64))
    completion = verify_current_provenance(
        launch, context="after sparse optimization before manifest commit"
    )
    manifest = {
        "study": STUDY, "protocol_version": PROTOCOL_VERSION,
        "run_tag": payload["run_tag"], "dataset": dataset, "session": session,
        "outer_subject": int(payload["outer_subject"]), "target_subject": target_subject,
        "role": role, "source_subjects": source_subjects,
        "source_subject_count": len(source_subjects), "seed": seed, "max_iter": max_iter,
        "checkpoint_iterations": list(CHECKPOINTS), "candidate_top_k": list(TOP_K),
        "selected_top_k": selected_top_k, "relevance_temperature": 1.0,
        "js_relevance_weight": js_weight,
        "selected_source_only_warmup_iters": selected_warmup,
        "model": "rwhedn_js", "rw_stage": "l1", "upstream_commit": UPSTREAM_COMMIT,
        "resolved_args": json_safe_namespace(args),
        "execution_contract": {
            "selection_checkpoint": 1000,
            "source_only_warmup_definition": (
                "iterations 1..N disable adversarial, target-cluster, target-center-update, "
                "and target-consistency terms while retaining source CE/source prototypes"
            ),
            "normalized_js_relevance_definition": (
                "weight=0 delegates to RWHEDN.forward; weight=1 uses "
                "z(source_CE)+z(JS(source_mean_probability,target_mean_probability))"
            ),
            "later_stage_e_terms_enabled": False,
            "top_k_changes_training": False,
            "target_labels_used_for_training_or_selection": False,
            "target_label_isolated": True,
            "target_labels_opened_after_probability_artifact": True,
            "target_labels_used_for_checkpoint_selection": False,
            "target_labels_used_for_early_stopping": False,
            "optimizer_process_boundary": "multiprocessing_spawn",
            "optimizer_child_received_target_labels": False,
            "optimizer_order": ["RMSprop_step1", "Adam_step2"],
            "source_reset_cluster_semantics": "stale_previous_batch",
            "target_reset_cluster_semantics": "refresh",
            "target_prediction_every_iteration": True,
        },
        "dbscan": {
            "eps": float(loader_builder.best_cluster_params["eps"]),
            "min_samples": int(loader_builder.best_cluster_params["min_samples"]),
            "retained_mask_sha256": sha256_array(retained_mask.astype(np.bool_)),
        },
        "input_provenance": payload["input_provenance"],
        "state_sha256": {
            "initial": initial_state, "post_source_prototype_initialization": initialized_state,
            "fixed_final": sha256_model_state(trainer.get_model_state()),
        },
        "environment": {
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
            "optimizer_child_pid": os.getpid(), "optimizer_parent_pid": int(payload["parent_pid"]),
        },
        "launch_provenance": launch, "completion_provenance": completion,
        "git": launch["git"], "code_sha256": launch["code_sha256"],
        "launch_argv": payload["launch_argv"],
        "label_vault_state_at_manifest": payload["label_vault_state"],
        "label_vault_transitions_at_manifest": payload["label_vault_transitions"],
    }
    atomic_json(manifest_path, manifest)
    receipt = {
        "study": STUDY, "protocol_version": PROTOCOL_VERSION,
        "run_tag": payload["run_tag"], "dataset": dataset, "session": session,
        "outer_subject": int(payload["outer_subject"]), "target_subject": target_subject,
        "role": role, "optimization_complete": True, "iterations_completed": max_iter,
        "source_iterator_resets": source_resets, "target_iterator_resets": target_resets,
        "training_seconds": float(time.time() - started),
        "max_dense_reconstruction_abs_diff": max_dense_reconstruction_diff,
        "target_labels_still_sealed": True,
        "target_labels_used_for_training_or_selection": False,
        "optimizer_process_boundary": "multiprocessing_spawn",
        "optimizer_child_pid": os.getpid(), "optimizer_parent_pid": int(payload["parent_pid"]),
        "optimizer_child_received_target_labels": False,
        "trajectory_sha256": sha256_file(trajectory_path),
        "training_log_sha256": sha256_file(log_path),
        "fit_manifest_sha256": sha256_file(manifest_path),
        "launch_provenance": launch, "completion_provenance": completion,
        "committed_at": utc_now(),
    }
    atomic_json(receipt_path, receipt)
    return {
        "trajectory_sha256": receipt["trajectory_sha256"],
        "receipt_sha256": sha256_file(receipt_path),
    }


def child_entry(payload: dict[str, Any], sender: Any) -> None:
    try:
        sender.send({"ok": True, "result": train_child(payload)})
    except BaseException as exc:
        try:
            sender.send({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})
        finally:
            sender.close()
        raise
    else:
        sender.close()


def validate_fit(
    fit_dir: Path, *, role: str, dataset: str, session: int, outer_subject: int,
    target_subject: int, run_tag: str, launch: dict[str, Any]
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
        actual_identity = (
            document.get("role"), document.get("dataset"), document.get("session"),
            document.get("outer_subject"), document.get("target_subject"), document.get("run_tag"),
        )
        if actual_identity != expected:
            raise RuntimeError(f"fit identity mismatch at {fit_dir}: {actual_identity}")
    if receipt.get("trajectory_sha256") != sha256_file(fit_dir / "trajectories.npz"):
        raise RuntimeError(f"trajectory hash mismatch at {fit_dir}")
    if receipt.get("training_log_sha256") != sha256_file(fit_dir / "training_log.csv"):
        raise RuntimeError(f"training log hash mismatch at {fit_dir}")
    if receipt.get("fit_manifest_sha256") != sha256_file(fit_dir / "fit_manifest.json"):
        raise RuntimeError(f"manifest hash mismatch at {fit_dir}")
    if float(manifest.get("js_relevance_weight", -1.0)) not in JS_WEIGHTS:
        raise RuntimeError(f"normalized-JS manifest value mismatch at {fit_dir}")
    if evaluation.get("js_relevance_weight") != manifest.get("js_relevance_weight"):
        raise RuntimeError(f"normalized-JS evaluation value mismatch at {fit_dir}")
    if evaluation.get("selected_source_only_warmup_iters") != manifest.get("selected_source_only_warmup_iters"):
        raise RuntimeError(f"frozen warm-up value mismatch at {fit_dir}")
    for name in ("trajectories.npz", "training_log.csv", "fit_manifest.json",
                 "optimization_receipt.json", "pipeline_probabilities.npz", "target_labels.npz"):
        if evaluation["artifact_sha256"].get(name) != sha256_file(fit_dir / name):
            raise RuntimeError(f"evaluation artifact hash mismatch for {name} at {fit_dir}")
    for document in (manifest, receipt, evaluation):
        if not same_launch_identity(document.get("launch_provenance", {}), launch):
            raise RuntimeError(f"fit launch provenance mismatch at {fit_dir}")
        if document.get("completion_provenance", {}).get("code_sha256") != launch["code_sha256"]:
            raise RuntimeError(f"fit completion code mismatch at {fit_dir}")
        if document.get("completion_provenance", {}).get("git") != launch["git"]:
            raise RuntimeError(f"fit completion Git mismatch at {fit_dir}")
    return True


def evaluate_committed_fit(
    *, fit_dir: Path, vault: TargetLabelVault, role: str, dataset: str, session: int,
    outer_subject: int, target_subject: int, run_tag: str, launch: dict[str, Any],
    js_relevance_weight: float, selected_source_only_warmup_iters: int,
    selected_top_k: int, selected_temperature: float,
    selected_gate: float, baseline_path: Path,
    lossless_selected_probability: bool = False,
) -> dict[str, Any]:
    optimization_files = set(FIT_FILES[:4])
    actual = {entry.name for entry in fit_dir.iterdir()}
    if actual != optimization_files:
        raise RuntimeError(
            f"optimizer artifact boundary mismatch at {fit_dir}: {sorted(actual)}"
        )
    trajectory_path = fit_dir / "trajectories.npz"
    log_path = fit_dir / "training_log.csv"
    manifest_path = fit_dir / "fit_manifest.json"
    receipt_path = fit_dir / "optimization_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trajectory_sha = sha256_file(trajectory_path)
    if receipt.get("trajectory_sha256") != trajectory_sha:
        raise RuntimeError("child trajectory receipt mismatch")
    if receipt.get("target_labels_still_sealed") is not True:
        raise RuntimeError("child label-vault receipt mismatch")
    if (
        not same_launch_identity(manifest.get("launch_provenance", {}), launch)
        or not same_launch_identity(receipt.get("launch_provenance", {}), launch)
    ):
        raise RuntimeError("child provenance mismatch")
    completion = verify_current_provenance(
        launch, context="after normalized-JS route commit before frozen-pipeline reconstruction"
    )
    if not baseline_path.is_file():
        raise RuntimeError(f"immutable baseline prerequisite missing: {baseline_path}")
    baseline, baseline_identity = baseline_probability(baseline_path)
    route, route_identity = route_probability(
        trajectory_path, selected_top_k, selected_temperature
    )
    assert_identity(baseline_identity, route_identity, f"warm-up {role} pipeline")
    selected_probability = normalize(
        (1.0 - selected_gate) * baseline + selected_gate * route
    )
    pipeline_path = fit_dir / "pipeline_probabilities.npz"
    atomic_npz(
        pipeline_path,
        baseline_probability=baseline.astype(np.float32),
        route_probability=route.astype(np.float32),
        selected_probability=selected_probability.astype(
            np.float64 if lossless_selected_probability else np.float32
        ),
        selected_top_k_requested=np.asarray([selected_top_k], dtype=np.int16),
        selected_temperature=np.asarray([selected_temperature], dtype=np.float64),
        selected_gate=np.asarray([selected_gate], dtype=np.float64),
        js_relevance_weight=np.asarray([js_relevance_weight], dtype=np.float64),
        selected_source_only_warmup_iters=np.asarray(
            [selected_source_only_warmup_iters], dtype=np.int32
        ),
        target_subject=route_identity["subject"].astype(np.int16),
        target_session=route_identity["session"].astype(np.int8),
        target_trial=route_identity["trial"].astype(np.int16),
        target_sample_within_trial=route_identity["sample"].astype(np.int32),
        target_original_dataset_index=route_identity["index"].astype(np.int32),
        retained_mask=route_identity["retained"].astype(bool),
    )
    pipeline_sha = sha256_file(pipeline_path)
    with np.load(pipeline_path, allow_pickle=False) as committed:
        committed_selected_probability = committed["selected_probability"].astype(
            np.float64
        )
    target_one_hot = vault.open("posthoc_evaluation", probability_sha256=pipeline_sha)
    y_true = target_one_hot.argmax(axis=1).astype(np.int16)
    with np.load(trajectory_path, allow_pickle=False) as artifact:
        original_index = artifact["target_original_dataset_index"].astype(np.int32)
        target_trial = artifact["target_trial"].astype(np.int16)
        within_trial = artifact["target_sample_within_trial"].astype(np.int32)
        selected_requested = int(artifact["selected_top_k_requested"][0])
    if len(y_true) != selected_probability.shape[0] or selected_requested != selected_top_k:
        raise RuntimeError("selected pipeline/label lattice mismatch")
    labels_path = fit_dir / "target_labels.npz"
    atomic_npz(
        labels_path, y_true=y_true, target_subject=np.full(len(y_true), target_subject, dtype=np.int16),
        target_session=np.full(len(y_true), session, dtype=np.int8), target_trial=target_trial,
        target_sample_within_trial=within_trial, target_original_dataset_index=original_index,
    )
    selected_metrics = multiclass_metrics(y_true, committed_selected_probability)
    baseline_metrics = multiclass_metrics(y_true, baseline)
    route_metrics = multiclass_metrics(y_true, route)
    evaluation = {
        "study": STUDY, "protocol_version": PROTOCOL_VERSION, "run_tag": run_tag,
        "role": role, "dataset": dataset, "session": session,
        "outer_subject": outer_subject, "target_subject": target_subject,
        "js_relevance_weight": js_relevance_weight,
        "selected_source_only_warmup_iters": selected_source_only_warmup_iters,
        "selected_top_k_requested": selected_top_k,
        "selected_temperature": selected_temperature,
        "selected_gate": selected_gate,
        "selected_probability_storage_dtype": str(
            committed_selected_probability.dtype
            if lossless_selected_probability else np.dtype(np.float32)
        ),
        "selected_metrics": selected_metrics,
        "baseline_metrics": baseline_metrics,
        "route_metrics": route_metrics,
        "baseline_sha256": sha256_file(baseline_path),
        "target_labels_used_for_training_or_selection": False,
        "target_label_isolated": True,
        "target_labels_opened_after_probability_artifact": True,
        "optimizer_child_received_target_labels": False,
        "label_vault_final_state": vault.state,
        "label_vault_transitions": vault.transitions,
        "target_labels_sha256": sha256_array(target_one_hot),
        "artifact_sha256": {
            "trajectories.npz": trajectory_sha,
            "training_log.csv": sha256_file(log_path),
            "fit_manifest.json": sha256_file(manifest_path),
            "optimization_receipt.json": sha256_file(receipt_path),
            "pipeline_probabilities.npz": pipeline_sha,
            "target_labels.npz": sha256_file(labels_path),
        },
        "launch_provenance": launch, "completion_provenance": completion,
    }
    atomic_json(fit_dir / "evaluation.json", evaluation)
    if not validate_fit(
        fit_dir, role=role, dataset=dataset, session=session,
        outer_subject=outer_subject, target_subject=target_subject,
        run_tag=run_tag, launch=launch,
    ):
        raise RuntimeError("final fit validation failed")
    return evaluation


def run_fit(
    *, fit_dir: Path, vault: TargetLabelVault, role: str, dataset: str, session: int,
    outer_subject: int, target_subject: int, run_tag: str, seed: int,
    source_data: np.ndarray, source_labels: np.ndarray, source_groups: np.ndarray,
    target_data: np.ndarray, target_groups: np.ndarray, target_original_index: np.ndarray,
    js_relevance_weight: float, selected_source_only_warmup_iters: int,
    selected_top_k: int, selected_temperature: float,
    selected_gate: float, baseline_path: Path, launch: dict[str, Any],
    lossless_selected_probability: bool = False,
) -> dict[str, Any]:
    if validate_fit(
        fit_dir, role=role, dataset=dataset, session=session,
        outer_subject=outer_subject, target_subject=target_subject,
        run_tag=run_tag, launch=launch,
    ):
        existing = json.loads((fit_dir / "evaluation.json").read_text(encoding="utf-8"))
        frozen = (
            existing.get("js_relevance_weight"),
            existing.get("selected_source_only_warmup_iters"),
            existing.get("selected_top_k_requested"),
            existing.get("selected_temperature"),
            existing.get("selected_gate"),
        )
        expected_frozen = (
            js_relevance_weight, selected_source_only_warmup_iters,
            selected_top_k, selected_temperature, selected_gate
        )
        if frozen != expected_frozen:
            raise RuntimeError(f"resumed fit frozen-pipeline mismatch: {fit_dir}")
        return existing
    fit_dir.mkdir(parents=True, exist_ok=True)
    if any(fit_dir.iterdir()):
        raise RuntimeError(f"incomplete or non-empty fit directory: {fit_dir}")
    payload = {
        "fit_dir": str(fit_dir), "run_tag": run_tag, "dataset": dataset,
        "session": session, "outer_subject": outer_subject, "target_subject": target_subject,
        "role": role, "seed": seed, "max_iter": 1000,
        "js_relevance_weight": float(js_relevance_weight),
        "selected_source_only_warmup_iters": int(selected_source_only_warmup_iters),
        "source_data": np.ascontiguousarray(source_data),
        "source_labels": np.ascontiguousarray(source_labels, dtype=np.float32),
        "source_groups": np.ascontiguousarray(source_groups, dtype=np.int16),
        "target_data": np.ascontiguousarray(target_data),
        "target_groups": np.ascontiguousarray(target_groups, dtype=np.int16),
        "target_original_index": np.ascontiguousarray(target_original_index, dtype=np.int32),
        "selected_top_k": int(selected_top_k),
        "input_provenance": {
            "source_data_sha256": sha256_array(source_data),
            "source_labels_sha256": sha256_array(source_labels),
            "source_groups_sha256": sha256_array(source_groups),
            "target_data_sha256": sha256_array(target_data),
            "target_groups_sha256": sha256_array(target_groups),
        },
        "launch_provenance": launch, "launch_argv": list(sys.argv),
        "parent_pid": os.getpid(), "label_vault_state": vault.state,
        "label_vault_transitions": vault.transitions,
    }
    context = get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=child_entry, args=(payload, sender),
        name=(f"js-{dataset}-s{session}-o{outer_subject}-t{target_subject}-"
              f"js{js_relevance_weight:g}-w{selected_source_only_warmup_iters}-{role}"),
    )
    process.daemon = True
    try:
        process.start()
        sender.close()
        del payload
        process.join()
        message = receiver.recv() if receiver.poll() else None
    finally:
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        receiver.close()
        try:
            sender.close()
        except OSError:
            pass
    if process.exitcode != 0 or not message or message.get("ok") is not True:
        detail = message.get("traceback", message.get("error")) if message else "no child receipt"
        raise RuntimeError(f"spawned normalized-JS optimizer failed exit={process.exitcode}: {detail}")
    return evaluate_committed_fit(
        fit_dir=fit_dir, vault=vault, role=role, dataset=dataset, session=session,
        outer_subject=outer_subject, target_subject=target_subject, run_tag=run_tag,
        launch=launch, js_relevance_weight=js_relevance_weight,
        selected_source_only_warmup_iters=selected_source_only_warmup_iters,
        selected_top_k=selected_top_k, selected_temperature=selected_temperature,
        selected_gate=selected_gate, baseline_path=baseline_path,
        lossless_selected_probability=lossless_selected_probability,
    )


def lower_tail(values: list[float]) -> tuple[float, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    cvar_count = max(1, int(np.ceil(0.20 * len(ordered))))
    return float(ordered[:cvar_count].mean()), float(np.quantile(ordered, 0.10))


def select_js(inner_evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for js_weight in JS_WEIGHTS:
        metrics = [
            row["selected_metrics"] for row in inner_evaluations
            if float(row["js_relevance_weight"]) == js_weight
        ]
        if len(metrics) != 14:
            raise RuntimeError(f"normalized-JS inner lattice incomplete for {js_weight}: {len(metrics)}/14")
        uars = [float(value["uar"]) for value in metrics]
        cvar, p10 = lower_tail(uars)
        rows.append({
            "js_relevance_weight": js_weight,
            "mean_uar": float(np.mean(uars)),
            "mean_accuracy": float(np.mean([value["accuracy"] for value in metrics])),
            "cvar20_uar": cvar, "p10_uar": p10,
            "mean_ece": float(np.mean([value["ece_equal_mass_10"] for value in metrics])),
        })
    reference = next(row for row in rows if row["js_relevance_weight"] == 0)
    tolerance = 1e-10
    for row in rows:
        row["guardrail_deltas_vs_dense"] = {
            "accuracy": row["mean_accuracy"] - reference["mean_accuracy"],
            "cvar20_uar": row["cvar20_uar"] - reference["cvar20_uar"],
            "p10_uar": row["p10_uar"] - reference["p10_uar"],
            "ece": row["mean_ece"] - reference["mean_ece"],
        }
        delta = row["guardrail_deltas_vs_dense"]
        row["eligible"] = (
            delta["accuracy"] >= -tolerance and delta["cvar20_uar"] >= -tolerance
            and delta["p10_uar"] >= -tolerance and delta["ece"] <= tolerance
        )
    eligible = [row for row in rows if row["eligible"]]
    winner = max(
        eligible,
        key=lambda row: (
            row["mean_uar"], row["mean_accuracy"], row["cvar20_uar"],
            row["p10_uar"], -row["mean_ece"], -row["js_relevance_weight"],
        ),
    )
    return {
        "selection_metric": "mean_inner_pseudotarget_uar_with_zero_js_guardrails",
        "guardrail_reference_js_relevance_weight": 0,
        "guardrail_tolerance": tolerance,
        "tie_break": ["mean_uar", "mean_accuracy", "cvar20_uar", "p10_uar", "lowest_mean_ece", "lower_js_weight"],
        "candidate_rows": rows,
        "selected_js_relevance_weight": float(winner["js_relevance_weight"]),
    }


def validate_complete_outer_fold(
    fold_dir: Path, *, run_tag: str, dataset: str, session: int, subject: int,
    launch: dict[str, Any]
) -> bool:
    if not fold_dir.exists() or not (fold_dir / "outer_result.json").exists():
        return False
    expected_dirs = {f"inner_target_{value:02d}" for value in range(1, 16) if value != subject}
    expected_dirs.add("outer_fit")
    actual_dirs = {entry.name for entry in fold_dir.iterdir() if entry.is_dir()}
    actual_files = {entry.name for entry in fold_dir.iterdir() if entry.is_file()}
    if actual_dirs != expected_dirs or actual_files != set(ROOT_FILES):
        raise RuntimeError(f"outer fold artifact lattice mismatch: {fold_dir}")
    for pseudo in range(1, 16):
        if pseudo == subject:
            continue
        inner_dir = fold_dir / f"inner_target_{pseudo:02d}"
        expected_js = {js_directory(value) for value in JS_WEIGHTS}
        if {path.name for path in inner_dir.iterdir()} != expected_js:
            raise RuntimeError(f"inner normalized-JS lattice mismatch: {inner_dir}")
        for js_weight in JS_WEIGHTS:
            validate_fit(
                inner_dir / js_directory(js_weight), role="inner", dataset=dataset,
                session=session, outer_subject=subject, target_subject=pseudo,
                run_tag=run_tag, launch=launch,
            )
    validate_fit(
        fold_dir / "outer_fit", role="outer", dataset=dataset, session=session,
        outer_subject=subject, target_subject=subject, run_tag=run_tag, launch=launch,
    )
    result = json.loads((fold_dir / "outer_result.json").read_text(encoding="utf-8"))
    if (result.get("run_tag"), result.get("dataset"), result.get("session"), result.get("outer_subject")) != (
        run_tag, dataset, session, subject
    ):
        raise RuntimeError("outer result identity mismatch")
    return True


def run_outer_fold(dataset: str, session: int, subject: int, run_tag: str, seed: int) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if run_tag not in ALLOWED_TAGS or dataset not in DATASET_CONTRACT:
        raise ValueError((run_tag, dataset))
    if session not in (1, 2, 3) or subject not in range(1, 16) or seed != 42:
        raise ValueError((session, subject, seed))
    launch = {
        "code_sha256": code_provenance(), "git": git_provenance(),
        "captured_at": utc_now(), "parent_pid": os.getpid(), "argv": list(sys.argv),
    }
    fold_dir = fold_directory(run_tag, dataset, session, subject)
    if validate_complete_outer_fold(
        fold_dir, run_tag=run_tag, dataset=dataset, session=session, subject=subject,
        launch=launch,
    ):
        print(f"complete immutable normalized-JS outer fold: {fold_dir}", flush=True)
        return fold_dir
    lock = claim_path(run_tag, dataset, session, subject)
    token = acquire_claim(lock, {
        "study": STUDY, "run_tag": run_tag, "dataset": dataset,
        "session": session, "outer_subject": subject,
    })
    live_child = False
    try:
        fold_dir.mkdir(parents=True, exist_ok=True)
        allowed_partial = {f"inner_target_{value:02d}" for value in range(1, 16) if value != subject}
        allowed_partial |= {"outer_fit", *ROOT_FILES}
        foreign = {entry.name for entry in fold_dir.iterdir()} - allowed_partial
        if foreign:
            raise RuntimeError(f"foreign outer-fold artifacts: {sorted(foreign)}")

        setup_seed(seed)
        args = build_args(dataset, session, "rwhedn", seed, 1000)
        loaded = get_dataset(args)
        data = np.asarray(loaded["data"])
        labels = np.asarray(loaded["labels"], dtype=np.float32)
        groups = np.asarray(loaded["groups"], dtype=np.int16)
        validate_dataset(dataset, session, data, labels, groups)
        sid = groups[:, 0]
        outer_mask = sid == subject
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
            raise RuntimeError("outer source lattice mismatch")
        frozen = frozen_pipeline(run_tag, dataset, session, subject)
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
            for js_weight in JS_WEIGHTS:
                inner_vault = TargetLabelVault(source_labels_all[inner_target_mask])
                evaluation = run_fit(
                    fit_dir=inner_dir / js_directory(js_weight), vault=inner_vault,
                    role="inner", dataset=dataset, session=session, outer_subject=subject,
                    target_subject=pseudo, run_tag=run_tag, seed=seed,
                    source_data=source_data_all[inner_source_mask],
                    source_labels=source_labels_all[inner_source_mask],
                    source_groups=source_groups_all[inner_source_mask],
                    target_data=source_data_all[inner_target_mask],
                    target_groups=source_groups_all[inner_target_mask],
                    target_original_index=np.flatnonzero(source_mask)[inner_target_mask].astype(np.int32),
                    js_relevance_weight=js_weight,
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
                    f"js_weight={js_weight:g} complete", flush=True
                )

        selection = {
            "study": STUDY, "protocol_version": PROTOCOL_VERSION, "run_tag": run_tag,
            "dataset": dataset, "session": session, "outer_subject": subject,
            "inner_pseudo_target_subjects": outer_sources,
            "outer_labels_opened": False,
            "js_candidates": list(JS_WEIGHTS),
            "fixed_final_iteration": 1000,
            "frozen_pipeline": frozen,
            **select_js(inner_evaluations),
            "launch_provenance": launch,
        }
        atomic_json(fold_dir / "selection.json", selection)
        selected_js = float(selection["selected_js_relevance_weight"])
        outer_vault = TargetLabelVault(outer_labels_hidden)
        outer_evaluation = run_fit(
            fit_dir=fold_dir / "outer_fit", vault=outer_vault, role="outer",
            dataset=dataset, session=session, outer_subject=subject,
            target_subject=subject, run_tag=run_tag, seed=seed,
            source_data=source_data_all, source_labels=source_labels_all,
            source_groups=source_groups_all, target_data=outer_data,
            target_groups=outer_groups, target_original_index=outer_original,
            js_relevance_weight=selected_js,
            selected_source_only_warmup_iters=frozen_warmup,
            selected_top_k=frozen_top_k,
            selected_temperature=frozen_temperature,
            selected_gate=frozen_gate,
            baseline_path=baseline_path_for(
                run_tag, dataset, session, subject, subject, "outer"
            ),
            launch=launch,
        )
        completion = verify_current_provenance(launch, context="before normalized-JS outer result commit")
        result = {
            "study": STUDY, "protocol_version": PROTOCOL_VERSION, "run_tag": run_tag,
            "dataset": dataset, "session": session, "outer_subject": subject,
            "outer_source_subjects": outer_sources, "inner_pseudo_target_subjects": outer_sources,
            "selected_js_relevance_weight": selected_js,
            "selected_source_only_warmup_iters": frozen_warmup,
            "selected_top_k_requested": frozen_top_k,
            "selected_temperature": frozen_temperature,
            "selected_gate": frozen_gate,
            "selection": selection, "selected_outer_metrics": outer_evaluation["selected_metrics"],
            "outer_baseline_metrics": outer_evaluation["baseline_metrics"],
            "outer_route_metrics": outer_evaluation["route_metrics"],
            "target_labels_used_for_training_or_selection": False,
            "target_label_isolated": True,
            "outer_labels_opened_after_selection_and_probability_commit": True,
            "launch_provenance": launch, "completion_provenance": completion,
        }
        atomic_json(fold_dir / "outer_result.json", result)
        ledger = {}
        for path in sorted(fold_dir.rglob("*")):
            if path.is_file() and path.name != "fold_ledger.json":
                ledger[str(path.relative_to(fold_dir)).replace("\\", "/")] = sha256_file(path)
        atomic_json(fold_dir / "fold_ledger.json", {
            "study": STUDY, "run_tag": run_tag, "dataset": dataset,
            "session": session, "outer_subject": subject, "sha256": ledger,
            "launch_provenance": launch, "completion_provenance": completion,
        })
        if not validate_complete_outer_fold(
            fold_dir, run_tag=run_tag, dataset=dataset, session=session,
            subject=subject, launch=launch,
        ):
            raise RuntimeError("completed outer fold failed final validation")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return fold_dir
    finally:
        if live_child:
            print(f"child remains live; retaining claim {lock}", file=sys.stderr, flush=True)
        else:
            release_claim(lock, token)


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
