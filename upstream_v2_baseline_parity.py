"""Exact upstream-HEDN baseline parity runner with two isolated estimands.

The optimization engine intentionally reproduces the historical HEDN trainer,
including its source-iterator stale-cluster reset behavior.  The two modes differ
only in their legal access to held-out-target labels:

``label_isolated``
    Runs exactly 1,000 iterations.  Target labels are vaulted, the target loader
    receives zeros, and every target probability is committed before the vault
    can be opened for post-hoc evaluation.

``upstream_target_assisted``
    Reproduces the historical per-iteration held-out-target accuracy, strict
    earliest-maximum checkpoint retention, and target-driven early stopping.

The probability artifact never contains labels.  ``evaluation.json`` is the
only completion marker; any proper subset of the expected files is an incomplete
fold and is never overwritten automatically.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import secrets
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from data_utils._get_dataset import get_dataset
from data_utils.HEDNLoader import HEDNLoader
from get_model_utils import get_model_utils
from seed_series_upstream_oracle import (
    DATASET_CONTRACT,
    UPSTREAM_COMMIT,
    build_args,
    predict_batched,
    validate_dataset,
    within_trial_index,
)
from utils.utils import setup_seed


HERE = Path(__file__).resolve().parent
STUDY = "upstream_v2_baseline_parity"
PROTOCOL_VERSION = "v1"
MODES = ("label_isolated", "upstream_target_assisted")
PAPER_TAGS = {
    "label_isolated": "paper_upstream_v2_baseline_parity_label_isolated_v1",
    "upstream_target_assisted": "paper_upstream_v2_baseline_parity_upstream_target_assisted_v1",
}
SMOKE_TAGS = {
    "label_isolated": "smoke_upstream_v2_baseline_parity_label_isolated_v1",
    "upstream_target_assisted": "smoke_upstream_v2_baseline_parity_upstream_target_assisted_v1",
}
ALLOWED_TAGS = {
    mode: {PAPER_TAGS[mode], SMOKE_TAGS[mode]}
    for mode in MODES
}
# Empty by default.  A separately sealed study may add prespecified seeds before
# calling ``run_fold``; ordinary baseline-parity invocations remain locked to 42.
ADDITIONAL_SEEDS: set[int] = set()
EXPECTED_FILES = (
    "probabilities.npz",
    "training_log.csv",
    "fold_manifest.json",
    "optimization_receipt.json",
    "target_labels.npz",
    "evaluation.json",
)
LOG_HEADER = (
    "cls_loss,transfer_loss,cons_loss,src_cluster_loss,tgt_cluster_loss,"
    "source_accuracy,target_accuracy,best_target_accuracy,easy_source,hard_source"
)


def canonical_label_flags(mode: str) -> dict[str, bool]:
    if mode not in MODES:
        raise ValueError(mode)
    isolated = mode == "label_isolated"
    return {
        "target_labels_used_for_training_or_selection": not isolated,
        "target_label_isolated": isolated,
        "target_labels_opened_after_probability_artifact": isolated,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _hash_state_value(digest: Any, prefix: str, value: Any) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            _hash_state_value(digest, f"{prefix}/{key}", value[key])
        return
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(prefix.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(array.shape).encode("utf-8"))
        digest.update(memoryview(array).cast("B"))
        return
    digest.update(prefix.encode("utf-8"))
    digest.update(repr(value).encode("utf-8"))


def sha256_model_state(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    _hash_state_value(digest, "state", state)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"immutable JSON differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary basename short: the locked result hierarchy is close
    # to the legacy Windows MAX_PATH boundary before an atomic suffix is added.
    tmp = path.with_name(f".j{os.getpid()}.tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite immutable NPZ: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".n{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def atomic_training_log(path: Path, values: np.ndarray) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite immutable training log: {path}")
    tmp = path.with_name(f".l{os.getpid()}.tmp")
    np.savetxt(tmp, values, delimiter=",", header=LOG_HEADER, comments="", fmt="%.8f")
    os.replace(tmp, path)


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
        "upstream_v2_baseline_parity.py",
        "seed_series_upstream_oracle.py",
        "config.py",
        "hedn.yaml",
        "get_model_utils.py",
        "models/__init__.py",
        "models/HEDN.py",
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


def multiclass_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
    prediction = probabilities.argmax(axis=1)
    classes = np.arange(probabilities.shape[1])
    one_hot = np.eye(probabilities.shape[1], dtype=float)[y_true]
    confidence = probabilities.max(axis=1)
    correct = (prediction == y_true).astype(float)
    order = np.argsort(confidence, kind="stable")
    ece = 0.0
    for indices in np.array_split(order, 10):
        if indices.size:
            ece += (indices.size / y_true.size) * abs(
                float(confidence[indices].mean()) - float(correct[indices].mean())
            )
    return {
        "accuracy": float(np.mean(prediction == y_true)),
        "uar": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(
            y_true, prediction, labels=classes, average="macro", zero_division=0
        )),
        "classwise_f1": f1_score(
            y_true, prediction, labels=classes, average=None, zero_division=0
        ).astype(float).tolist(),
        "brier": float(np.mean(np.square(probabilities - one_hot).sum(axis=1))),
        "ece_equal_mass_10": float(ece),
        "confidence_mean": float(confidence.mean()),
        "confusion_matrix": confusion_matrix(
            y_true, prediction, labels=classes
        ).astype(int).tolist(),
    }


class TargetLabelVault:
    """Minimal state machine preventing optimization code from receiving labels."""

    def __init__(self, one_hot_labels: np.ndarray):
        self.__payload = np.ascontiguousarray(one_hot_labels, dtype=np.float32).copy()
        self._state = "sealed"
        self._transitions: list[dict[str, Any]] = [{
            "sequence": 0,
            "state": "sealed",
            "at": utc_now(),
        }]

    @property
    def state(self) -> str:
        return self._state

    @property
    def transitions(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._transitions)

    def open(self, reason: str, probability_sha256: str | None = None) -> np.ndarray:
        if self._state == "sealed":
            if reason == "posthoc_evaluation" and not probability_sha256:
                raise RuntimeError("post-hoc label opening requires a committed probability hash")
            self._state = (
                "labels_opened_for_posthoc_evaluation"
                if reason == "posthoc_evaluation"
                else "labels_opened_for_target_assisted_optimization"
            )
            self._transitions.append({
                "sequence": len(self._transitions),
                "state": self._state,
                "reason": reason,
                "probability_sha256": probability_sha256,
                "at": utc_now(),
            })
        return self.__payload.copy()


def output_directory(
    run_tag: str, dataset: str, session: int, mode: str, subject: int
) -> Path:
    return (
        HERE / "results" / STUDY / run_tag / dataset / f"session_{session}" /
        mode / f"target_subject_{subject:02d}"
    )


def claim_path_for(
    run_tag: str, dataset: str, session: int, mode: str, subject: int
) -> Path:
    identity = f"{run_tag}|{dataset}|{session}|{mode}|{subject}"
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return HERE / "results" / STUDY / ".claims" / f"{key}.lock"


def acquire_fold_claim(
    *,
    run_tag: str,
    dataset: str,
    session: int,
    subject: int,
    mode: str,
    launch_code: dict[str, Any],
    launch_git: dict[str, Any],
) -> tuple[Path, str]:
    """Atomically claim an incomplete fold."""
    path = claim_path_for(run_tag, dataset, session, mode, subject)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    payload = {
        "study": STUDY,
        "protocol_version": PROTOCOL_VERSION,
        "run_tag": run_tag,
        "dataset": dataset,
        "session": session,
        "target_subject": subject,
        "mode": mode,
        "pid": os.getpid(),
        "token": token,
        "claimed_at": utc_now(),
        "launch_code_combined_sha256": launch_code["combined_sha256"],
        "launch_git": launch_git,
    }
    raw = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"fold claim already exists: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path, token


def release_fold_claim(path: Path, token: str) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot verify fold claim before release: {path}") from exc
    if payload.get("token") != token or payload.get("pid") != os.getpid():
        raise RuntimeError(f"fold claim ownership changed; refusing release: {path}")
    path.unlink()


def validate_completed_fold(
    fold_dir: Path,
    *,
    run_tag: str,
    dataset: str,
    session: int,
    subject: int,
    mode: str,
    max_iter: int,
    launch_code: dict[str, Any],
    launch_git: dict[str, Any],
) -> bool:
    if not fold_dir.exists():
        return False
    if not fold_dir.is_dir():
        raise RuntimeError(f"fold output path is not a directory: {fold_dir}")
    entries = list(fold_dir.iterdir())
    if not entries:
        return False
    actual = {entry.name for entry in entries}
    expected_files = set(EXPECTED_FILES)
    if actual != expected_files or any(not entry.is_file() for entry in entries):
        missing = sorted(expected_files - actual)
        foreign = sorted(actual - expected_files)
        raise RuntimeError(
            f"partial or unexpected fold artifacts found: {fold_dir} "
            f"(missing={missing}, foreign={foreign})"
        )

    evaluation = json.loads((fold_dir / "evaluation.json").read_text(encoding="utf-8"))
    manifest = json.loads((fold_dir / "fold_manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((fold_dir / "optimization_receipt.json").read_text(encoding="utf-8"))
    expected = {
        "study": STUDY,
        "protocol_version": PROTOCOL_VERSION,
        "run_tag": run_tag,
        "dataset": dataset,
        "session": session,
        "target_subject": subject,
        "mode": mode,
        "max_iter": max_iter,
    }
    for artifact_name, payload in (
        ("evaluation", evaluation),
        ("manifest", manifest),
        ("receipt", receipt),
    ):
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"completed fold {artifact_name} identity mismatch {key}: {fold_dir}"
                )
    if manifest.get("source_subject_count") != 14 or evaluation.get("source_subject_count") != 14:
        raise RuntimeError(f"completed fold source-count mismatch: {fold_dir}")
    if receipt.get("optimization_complete") is not True or receipt.get("probabilities_committed") is not True:
        raise RuntimeError(f"completed fold optimization receipt is not committed: {fold_dir}")

    hashes = evaluation.get("artifact_sha256", {})
    for name in EXPECTED_FILES[:-1]:
        if hashes.get(name) != sha256_file(fold_dir / name):
            raise RuntimeError(f"completed fold hash mismatch {name}: {fold_dir}")
    if receipt.get("probability_sha256") != hashes.get("probabilities.npz"):
        raise RuntimeError(f"completed fold probability receipt mismatch: {fold_dir}")
    if receipt.get("training_log_sha256") != hashes.get("training_log.csv"):
        raise RuntimeError(f"completed fold log receipt mismatch: {fold_dir}")
    if receipt.get("fold_manifest_sha256") != hashes.get("fold_manifest.json"):
        raise RuntimeError(f"completed fold manifest receipt mismatch: {fold_dir}")

    expected_flags = canonical_label_flags(mode)
    for artifact_name, payload in (
        ("manifest", manifest.get("execution_contract", {})),
        ("receipt", receipt),
        ("evaluation", evaluation),
    ):
        for key, value in expected_flags.items():
            if payload.get(key) is not value:
                raise RuntimeError(
                    f"completed fold {artifact_name} protocol flag mismatch {key}: {fold_dir}"
                )
    execution = manifest.get("execution_contract", {})
    target_access = mode == "upstream_target_assisted"
    expected_execution = {
        "target_loader_real_labels": target_access,
        "target_labels_used_in_loss": False,
        "target_labels_used_for_checkpoint_selection": target_access,
        "target_labels_used_for_early_stopping": target_access,
        "optimizer_process_boundary": "multiprocessing_spawn",
        "optimizer_child_received_target_labels": target_access,
    }
    expected_evaluation = {
        "target_labels_available_to_optimization": target_access,
        "target_labels_used_in_loss": False,
        "target_labels_used_for_checkpoint_selection": target_access,
        "target_labels_used_for_early_stopping": target_access,
        "optimizer_process_boundary": "multiprocessing_spawn",
        "optimizer_child_received_target_labels": target_access,
    }
    expected_receipt = {
        "target_labels_opened_during_optimization": target_access,
        "target_labels_still_sealed": not target_access,
        "optimizer_process_boundary": "multiprocessing_spawn",
        "optimizer_child_received_target_labels": target_access,
    }
    for artifact_name, payload, contract in (
        ("manifest", execution, expected_execution),
        ("evaluation", evaluation, expected_evaluation),
        ("receipt", receipt, expected_receipt),
    ):
        for key, value in contract.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"completed fold {artifact_name} execution mismatch {key}: {fold_dir}"
                )

    if manifest.get("code_sha256") != launch_code:
        raise RuntimeError(f"completed fold code provenance is not current: {fold_dir}")
    if evaluation.get("code_combined_sha256") != launch_code.get("combined_sha256"):
        raise RuntimeError(f"completed fold evaluation code digest is not current: {fold_dir}")
    for artifact_name, payload in (("manifest", manifest), ("receipt", receipt), ("evaluation", evaluation)):
        launch = payload.get("launch_provenance", {})
        completion = payload.get("completion_provenance", {})
        if launch.get("code_sha256") != launch_code or launch.get("git") != launch_git:
            raise RuntimeError(f"completed fold {artifact_name} launch provenance mismatch: {fold_dir}")
        if completion.get("code_sha256") != launch_code or completion.get("git") != launch_git:
            raise RuntimeError(f"completed fold {artifact_name} completion provenance mismatch: {fold_dir}")
    if manifest.get("git") != launch_git:
        raise RuntimeError(f"completed fold Git provenance is not current: {fold_dir}")
    return True


@torch.no_grad()
def evaluate_retained_probabilities(model: Any, retained_features: torch.Tensor, device: Any) -> np.ndarray:
    model.eval()
    probabilities = model.predict_proba(retained_features.to(device), mode="target")
    return probabilities.detach().cpu().numpy().astype(np.float32, copy=False)


def train_exact_upstream(
    trainer: Any,
    source_loader: Any,
    target_loader: Any,
    *,
    mode: str,
    target_y_true: np.ndarray | None,
) -> dict[str, Any]:
    if mode == "label_isolated" and target_y_true is not None:
        raise RuntimeError("label-isolated optimization must not receive target labels")
    if mode == "upstream_target_assisted" and target_y_true is None:
        raise RuntimeError("target-assisted optimization requires target labels")

    initial_state_sha256 = sha256_model_state(trainer.get_model_state())
    trainer.pre_training_processing(source_loader)
    post_initialization_state_sha256 = sha256_model_state(trainer.get_model_state())

    # Exact upstream iterator construction order.
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    retained_features = target_loader.dataset.d1
    trajectory: list[np.ndarray] = []
    log: list[list[float]] = []
    best_acc = 0.0
    best_iteration: int | None = None
    source_resets = 0
    target_resets = 0
    stopping_reason = "max_iter"

    for it in range(trainer.max_iter):
        trainer.model.train()
        try:
            src_data, src_label, src_cluster = next(source_iter)
        except StopIteration:
            source_resets += 1
            source_iter = iter(source_loader)
            # Intentional upstream parity: discard the refreshed cluster tensor
            # and retain src_cluster from the preceding batch.
            src_data, src_label, _ = next(source_iter)
        try:
            tgt_data, _, tgt_cluster = next(target_iter)
        except StopIteration:
            target_resets += 1
            target_iter = iter(target_loader)
            tgt_data, _, tgt_cluster = next(target_iter)

        src_data = src_data.to(trainer.device)
        src_label = src_label.to(trainer.device)
        tgt_data = tgt_data.to(trainer.device)

        cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss, easy_idx, hard_idx = (
            trainer.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
        )
        loss = (
            cls_loss
            + trainer.transfer_loss_weight * transfer_loss
            + trainer.constraint_loss_weight * cons_loss
        )
        trainer.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        trainer.optimizer.step()

        cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss, easy_idx, hard_idx = (
            trainer.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
        )
        trainer.fe_opt.zero_grad()
        (src_clu_loss + tgt_clu_loss).backward()
        trainer.fe_opt.step()

        # Exact upstream evaluation order.  The target prediction is still
        # executed in label-isolated mode to preserve train/eval transitions.
        source_acc = trainer.test(source_loader, mode="source")
        retained_probs = evaluate_retained_probabilities(
            trainer.model, retained_features, trainer.device
        )
        trajectory.append(retained_probs)

        if mode == "upstream_target_assisted":
            target_acc = float(
                100.0 * np.mean(retained_probs.argmax(axis=1) == target_y_true)
            )
            previous = best_acc
            best_acc = trainer._update_best_model(target_acc, best_acc)
            if best_acc > previous:
                best_iteration = it + 1
        else:
            target_acc = float("nan")

        iter_losses = {
            "cls_loss": cls_loss.detach().item() if isinstance(cls_loss, torch.Tensor) else cls_loss,
            "transfer_loss": transfer_loss.detach().item() if isinstance(transfer_loss, torch.Tensor) else transfer_loss,
            "cons_loss": cons_loss.detach().item() if isinstance(cons_loss, torch.Tensor) else cons_loss,
            "src_clu_loss": src_clu_loss.detach().item() if isinstance(src_clu_loss, torch.Tensor) else src_clu_loss,
            "tgt_clu_loss": tgt_clu_loss.detach().item() if isinstance(tgt_clu_loss, torch.Tensor) else tgt_clu_loss,
        }
        log.append([
            *[float(value) for value in iter_losses.values()],
            float(source_acc),
            float(target_acc),
            float(best_acc) if mode == "upstream_target_assisted" else float("nan"),
            float(int(easy_idx.cpu().numpy()) + 1),
            float(int(hard_idx.cpu().numpy()) + 1),
        ])
        trainer._log_training_info(it, iter_losses, source_acc, target_acc, best_acc)

        if mode == "upstream_target_assisted" and trainer._should_early_stop(best_acc):
            stopping_reason = (
                "perfect_target_accuracy"
                if 100.0 - best_acc < 1e-3
                else "target_accuracy_patience"
            )
            break

    return {
        "training_log": np.asarray(log, dtype=np.float64),
        "retained_probability_trajectory": np.stack(trajectory).astype(np.float32),
        "best_target_accuracy_percent": (
            float(best_acc) if mode == "upstream_target_assisted" else None
        ),
        "best_iteration": best_iteration,
        "iterations_completed": len(log),
        "stopping_reason": stopping_reason,
        "source_iterator_resets": source_resets,
        "target_iterator_resets": target_resets,
        "initial_state_sha256": initial_state_sha256,
        "post_initialization_state_sha256": post_initialization_state_sha256,
        "final_state_sha256": sha256_model_state(trainer.get_model_state()),
    }


def verify_current_provenance(
    launch_provenance: dict[str, Any], *, context: str
) -> dict[str, Any]:
    current_code = code_provenance()
    current_git = git_provenance()
    if current_code != launch_provenance["code_sha256"]:
        raise RuntimeError(f"code provenance changed {context}")
    if current_git != launch_provenance["git"]:
        raise RuntimeError(f"Git provenance changed {context}")
    return {
        "code_sha256": current_code,
        "git": current_git,
        "checked_at": utc_now(),
        "context": context,
    }


def optimizer_child(payload: dict[str, Any]) -> dict[str, Any]:
    """Run optimization in a spawned interpreter and commit label-free artifacts."""
    mode = str(payload["mode"])
    dataset = str(payload["dataset"])
    session = int(payload["session"])
    subject = int(payload["subject"])
    seed = int(payload["seed"])
    max_iter = int(payload["max_iter"])
    run_tag = str(payload["run_tag"])
    fold_dir = Path(payload["fold_dir"])
    launch_provenance = payload["launch_provenance"]

    common_payload_keys = {
        "fold_dir", "run_tag", "mode", "dataset", "session", "subject",
        "seed", "max_iter", "source_data", "source_labels", "source_groups",
        "target_data", "target_groups", "target_original_index",
        "input_provenance", "launch_provenance", "launch_argv", "parent_pid",
        "label_vault_state", "label_vault_transitions",
    }
    expected_payload_keys = set(common_payload_keys)
    if mode == "upstream_target_assisted":
        expected_payload_keys.add("target_labels")
    if set(payload) != expected_payload_keys:
        raise RuntimeError(
            "spawned optimizer payload key contract mismatch: "
            f"missing={sorted(expected_payload_keys - set(payload))}, "
            f"foreign={sorted(set(payload) - expected_payload_keys)}"
        )
    expected_input_hashes = {
        "data_sha256", "groups_sha256", "source_labels_sha256",
        "target_data_sha256", "target_groups_sha256",
    }
    if set(payload["input_provenance"]) != expected_input_hashes:
        raise RuntimeError("spawned optimizer input-provenance key contract mismatch")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required in the spawned optimizer process")
    if mode == "label_isolated" and "target_labels" in payload:
        raise RuntimeError("label-isolated optimizer payload contains target labels")
    if mode == "upstream_target_assisted" and "target_labels" not in payload:
        raise RuntimeError("target-assisted optimizer payload lacks target labels")
    expected_vault_state = (
        "sealed"
        if mode == "label_isolated"
        else "labels_opened_for_target_assisted_optimization"
    )
    if payload["label_vault_state"] != expected_vault_state:
        raise RuntimeError("spawned optimizer vault-state contract mismatch")
    if any(fold_dir.iterdir()):
        raise RuntimeError(f"optimizer child requires an empty fold directory: {fold_dir}")
    verify_current_provenance(
        launch_provenance, context="before spawned optimizer construction"
    )

    # Reproduce the upstream pre-dataset seed point inside this fresh process;
    # the parent dataset load is deterministic and cannot transfer RNG state
    # across the spawn boundary.
    setup_seed(seed)
    args = build_args(dataset, session, "hedn", seed, max_iter)
    args.early_stop = max_iter if mode == "upstream_target_assisted" else 0
    source_data = np.ascontiguousarray(payload["source_data"])
    source_labels = np.ascontiguousarray(payload["source_labels"], dtype=np.float32)
    source_groups = np.ascontiguousarray(payload["source_groups"], dtype=np.int16)
    target_data = np.ascontiguousarray(payload["target_data"])
    target_groups = np.ascontiguousarray(payload["target_groups"], dtype=np.int16)
    source_subjects = sorted(np.unique(source_groups[:, 0]).astype(int).tolist())
    if len(source_subjects) != 14 or subject in source_subjects:
        raise RuntimeError("spawned optimizer LOSO source identity contract failed")

    if mode == "upstream_target_assisted":
        target_labels_for_loader = np.ascontiguousarray(
            payload["target_labels"], dtype=np.float32
        )
        target_y_for_optimization: np.ndarray | None = (
            target_labels_for_loader.argmax(axis=1).astype(np.int16)
        )
    else:
        target_labels_for_loader = np.zeros(
            (len(target_data), DATASET_CONTRACT[dataset]["num_classes"]),
            dtype=np.float32,
        )
        target_y_for_optimization = None
    train_dataset = {
        "data": source_data,
        "labels": source_labels,
        "groups": source_groups,
    }
    target_dataset = {
        "data": target_data,
        "labels": target_labels_for_loader,
        "groups": target_groups,
    }

    # This reset is the verified upstream reset immediately before loader/model
    # construction.  Because spawn starts a fresh interpreter, no parent RNG or
    # label-vault state is inherited by the optimizer.
    setup_seed(seed)
    loader_builder = HEDNLoader(args, train_dataset, target_dataset)
    source_loader, target_loader = loader_builder()
    cluster_params = {
        "eps": float(loader_builder.best_cluster_params["eps"]),
        "min_samples": int(loader_builder.best_cluster_params["min_samples"]),
    }
    retained_mask = DBSCAN(**cluster_params).fit(target_data).labels_ != -1
    if int(retained_mask.sum()) != len(target_loader.dataset):
        raise RuntimeError("target retained-mask mismatch")
    if mode == "upstream_target_assisted":
        target_y_for_optimization = target_y_for_optimization[retained_mask]
    elif np.any(target_loader.dataset.d2.numpy()):
        raise RuntimeError("label-isolated target loader contains nonzero labels")

    # Loader construction mutates cluster dimensions; model and optimizers are
    # constructed afterward, preserving the exact historical call order.
    trainer = get_model_utils(args)
    started = time.time()
    trained = train_exact_upstream(
        trainer,
        source_loader,
        target_loader,
        mode=mode,
        target_y_true=target_y_for_optimization,
    )
    training_seconds = time.time() - started

    retained_x = target_loader.dataset.d1
    retained_last_batched_probs = predict_batched(trainer.model, retained_x)
    all_last_batched_probs = predict_batched(trainer.model, target_data)
    final_state = copy.deepcopy(trainer.get_model_state())
    if mode == "upstream_target_assisted":
        best_state = trainer.get_best_model_state()
        if best_state is None:
            raise RuntimeError("target-assisted mode did not retain a best state")
        trainer.model.load_state(best_state)
        retained_best_batched_probs = predict_batched(trainer.model, retained_x)
        all_best_batched_probs = predict_batched(trainer.model, target_data)
        best_state_sha256 = sha256_model_state(best_state)
        trainer.model.load_state(final_state)
    else:
        retained_best_batched_probs = np.empty(
            (0, DATASET_CONTRACT[dataset]["num_classes"]), dtype=np.float32
        )
        all_best_batched_probs = retained_best_batched_probs.copy()
        best_state_sha256 = None

    trials = target_groups[:, 1].astype(np.int16)
    sample_within_trial = within_trial_index(trials)
    original_index = np.asarray(payload["target_original_index"], dtype=np.int32)
    retained_index = np.flatnonzero(retained_mask).astype(np.int32)
    probability_path = fold_dir / "probabilities.npz"
    log_path = fold_dir / "training_log.csv"
    manifest_path = fold_dir / "fold_manifest.json"
    receipt_path = fold_dir / "optimization_receipt.json"

    atomic_npz(
        probability_path,
        iteration=np.arange(1, trained["iterations_completed"] + 1, dtype=np.int32),
        retained_probability_trajectory=trained["retained_probability_trajectory"],
        retained_last_eval_probs=trained["retained_probability_trajectory"][-1],
        retained_last_batched_probs=retained_last_batched_probs,
        all_last_batched_probs=all_last_batched_probs,
        retained_best_batched_probs=retained_best_batched_probs,
        all_best_batched_probs=all_best_batched_probs,
        retained_mask=retained_mask.astype(np.bool_),
        retained_target_local_index=retained_index,
        all_subject=np.full(len(target_data), subject, dtype=np.int16),
        all_session=np.full(len(target_data), session, dtype=np.int8),
        all_trial=trials,
        all_sample_within_trial=sample_within_trial,
        all_original_dataset_index=original_index,
    )
    atomic_training_log(log_path, trained["training_log"])

    completion_provenance = verify_current_provenance(
        launch_provenance, context="after optimization before manifest commit"
    )
    source_path = args.seed3_path if dataset == "seed3" else args.seed4_path
    execution_contract = {
        "loader_order": ["source", "target"],
        "optimizer_order": ["RMSprop_step1", "Adam_step2"],
        "per_iteration_order": [
            "train_mode", "source_next", "target_next", "RMSprop_phase",
            "Adam_phase", "source_eval", "target_eval",
        ],
        "source_reset_cluster_semantics": "stale_previous_batch",
        "target_reset_cluster_semantics": "refresh",
        "target_prediction_every_iteration": True,
        "target_loader_real_labels": mode == "upstream_target_assisted",
        "target_labels_used_in_loss": False,
        "target_labels_used_for_checkpoint_selection": mode == "upstream_target_assisted",
        "target_labels_used_for_early_stopping": mode == "upstream_target_assisted",
        "optimizer_process_boundary": "multiprocessing_spawn",
        "optimizer_child_received_target_labels": mode == "upstream_target_assisted",
        **canonical_label_flags(mode),
        "reported_endpoint": (
            "iteration_1000_fixed_final"
            if mode == "label_isolated"
            else "maximum_held_out_target_accuracy_earliest_tie"
        ),
    }
    manifest = {
        "study": STUDY,
        "protocol_version": PROTOCOL_VERSION,
        "run_tag": run_tag,
        "mode": mode,
        "dataset": dataset,
        "dataset_label": DATASET_CONTRACT[dataset]["label"],
        "session": session,
        "target_subject": subject,
        "source_subjects": source_subjects,
        "source_subject_count": 14,
        "seed": seed,
        "max_iter": max_iter,
        "upstream_commit": UPSTREAM_COMMIT,
        "resolved_args": json_safe_namespace(args),
        "dbscan": {
            **cluster_params,
            "selected_from_first_legal_source_subject_labels": True,
            "retained_mask_sha256": sha256_array(retained_mask.astype(np.bool_)),
        },
        "execution_contract": execution_contract,
        "input_provenance": {
            **payload["input_provenance"],
            "configured_source_path": str(Path(source_path).resolve()),
        },
        "state_sha256": {
            "initial": trained["initial_state_sha256"],
            "post_source_prototype_initialization": trained["post_initialization_state_sha256"],
            "last_at_stop": trained["final_state_sha256"],
            "target_best": best_state_sha256,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "optimizer_process_start_method": "spawn",
            "optimizer_child_pid": os.getpid(),
            "optimizer_parent_pid": int(payload["parent_pid"]),
        },
        "git": launch_provenance["git"],
        "code_sha256": launch_provenance["code_sha256"],
        "launch_provenance": launch_provenance,
        "completion_provenance": completion_provenance,
        "launch_argv": payload["launch_argv"],
        "label_vault_state_at_manifest": payload["label_vault_state"],
        "label_vault_transitions_at_manifest": payload["label_vault_transitions"],
    }
    atomic_json(manifest_path, manifest)

    probability_sha = sha256_file(probability_path)
    training_summary = {
        "iterations_completed": int(trained["iterations_completed"]),
        "best_iteration": trained["best_iteration"],
        "best_target_accuracy_percent": trained["best_target_accuracy_percent"],
        "stopping_reason": trained["stopping_reason"],
        "source_iterator_resets": int(trained["source_iterator_resets"]),
        "target_iterator_resets": int(trained["target_iterator_resets"]),
        "training_seconds": float(training_seconds),
    }
    receipt = {
        "study": STUDY,
        "protocol_version": PROTOCOL_VERSION,
        "run_tag": run_tag,
        "mode": mode,
        "dataset": dataset,
        "session": session,
        "target_subject": subject,
        "seed": seed,
        "max_iter": max_iter,
        "optimization_complete": True,
        "probabilities_committed": True,
        "iterations_completed": trained["iterations_completed"],
        "training_summary": training_summary,
        "optimizer_process_boundary": "multiprocessing_spawn",
        "optimizer_child_pid": os.getpid(),
        "optimizer_parent_pid": int(payload["parent_pid"]),
        "optimizer_child_received_target_labels": mode == "upstream_target_assisted",
        "target_labels_opened_during_optimization": mode == "upstream_target_assisted",
        "target_labels_still_sealed": mode == "label_isolated",
        **canonical_label_flags(mode),
        "probability_sha256": probability_sha,
        "training_log_sha256": sha256_file(log_path),
        "fold_manifest_sha256": sha256_file(manifest_path),
        "launch_provenance": launch_provenance,
        "completion_provenance": completion_provenance,
        "committed_at": utc_now(),
    }
    atomic_json(receipt_path, receipt)
    return {
        "optimization_complete": True,
        "probability_sha256": probability_sha,
        "receipt_sha256": sha256_file(receipt_path),
    }


def optimizer_child_entry(payload: dict[str, Any], sender: Any) -> None:
    try:
        sender.send({"ok": True, "result": optimizer_child(payload)})
    except BaseException as exc:
        try:
            sender.send({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
        finally:
            sender.close()
        raise
    else:
        sender.close()


def run_fold(
    *,
    dataset: str,
    session: int,
    subject: int,
    mode: str,
    seed: int,
    max_iter: int,
    run_tag: str,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for protocol parity")
    if mode not in MODES or run_tag not in ALLOWED_TAGS[mode]:
        raise ValueError(f"illegal mode/tag pairing: {mode}/{run_tag}")
    if dataset not in DATASET_CONTRACT or session not in (1, 2, 3) or subject not in range(1, 16):
        raise ValueError((dataset, session, subject))
    if seed != 42 or max_iter != 1000:
        raise ValueError("baseline parity requires seed 42 and max_iter=1000")
    launch_code = code_provenance()
    launch_git = git_provenance()
    manifest_provenance = {
        "study": STUDY,
        "protocol_version": PROTOCOL_VERSION,
        "run_tag": run_tag,
        "dataset": dataset,
        "session": session,
        "mode": mode,
        "target_subject": subject,
        "code_sha256": launch_code,
        "git": launch_git,
        "captured_at": utc_now(),
        "parent_pid": os.getpid(),
        "argv": list(sys.argv),
    }
    launch_provenance = manifest_provenance
    fold_dir = output_directory(run_tag, dataset, session, mode, subject)
    preexisting_claim = claim_path_for(run_tag, dataset, session, mode, subject)
    if preexisting_claim.exists():
        raise RuntimeError(
            f"fold claim already exists: {preexisting_claim}"
        )
    validation_args = {
        "run_tag": run_tag,
        "dataset": dataset,
        "session": session,
        "subject": subject,
        "mode": mode,
        "max_iter": max_iter,
        "launch_code": launch_code,
        "launch_git": launch_git,
    }
    if validate_completed_fold(fold_dir, **validation_args):
        print(f"complete immutable parity fold: {fold_dir}", flush=True)
        return fold_dir

    claim_path, claim_token = acquire_fold_claim(
        run_tag=run_tag,
        dataset=dataset,
        session=session,
        subject=subject,
        mode=mode,
        launch_code=launch_code,
        launch_git=launch_git,
    )
    process: Any | None = None
    try:
        # Close the check/claim race without ever placing lock files inside the
        # immutable artifact directory.
        if validate_completed_fold(fold_dir, **validation_args):
            print(f"complete immutable parity fold: {fold_dir}", flush=True)
            return fold_dir
        fold_dir.mkdir(parents=True, exist_ok=True)
        if any(fold_dir.iterdir()):
            raise RuntimeError(f"fold directory is not empty after claim: {fold_dir}")

        # Match upstream seeding around deterministic dataset construction.  A
        # separate reset occurs inside the spawned optimizer before its loader.
        setup_seed(seed)
        args = build_args(dataset, session, "hedn", seed, max_iter)
        args.early_stop = max_iter if mode == "upstream_target_assisted" else 0
        loaded = get_dataset(args)
        data = np.asarray(loaded["data"])
        labels = np.asarray(loaded["labels"], dtype=np.float32)
        groups = np.asarray(loaded["groups"], dtype=np.int16)
        validate_dataset(dataset, session, data, labels, groups)

        target_mask = groups[:, 0] == subject
        source_mask = ~target_mask
        source_groups = np.ascontiguousarray(groups[source_mask])
        source_subjects = sorted(np.unique(source_groups[:, 0]).astype(int).tolist())
        if len(source_subjects) != 14 or subject in source_subjects:
            raise RuntimeError("LOSO source identity contract failed")
        source_data = np.ascontiguousarray(data[source_mask])
        source_labels = np.ascontiguousarray(labels[source_mask])
        target_data = np.ascontiguousarray(data[target_mask])
        target_groups = np.ascontiguousarray(groups[target_mask])
        target_original_index = np.flatnonzero(target_mask).astype(np.int32)
        vault = TargetLabelVault(labels[target_mask])
        input_provenance = {
            "data_sha256": sha256_array(data),
            "groups_sha256": sha256_array(groups),
            "source_labels_sha256": sha256_array(source_labels),
            "target_data_sha256": sha256_array(target_data),
            "target_groups_sha256": sha256_array(target_groups),
        }

        child_payload: dict[str, Any] = {
            "fold_dir": str(fold_dir),
            "run_tag": run_tag,
            "mode": mode,
            "dataset": dataset,
            "session": session,
            "subject": subject,
            "seed": seed,
            "max_iter": max_iter,
            "source_data": source_data,
            "source_labels": source_labels,
            "source_groups": source_groups,
            "target_data": target_data,
            "target_groups": target_groups,
            "target_original_index": target_original_index,
            "input_provenance": input_provenance,
            "launch_provenance": launch_provenance,
            "launch_argv": list(sys.argv),
            "parent_pid": os.getpid(),
        }
        if mode == "upstream_target_assisted":
            child_payload["target_labels"] = vault.open(
                "target_assisted_optimization"
            )
        child_payload["label_vault_state"] = vault.state
        child_payload["label_vault_transitions"] = vault.transitions

        # Ensure the large all-label array cannot be inherited or serialized;
        # the isolated child payload has no target-label field at all.
        labels[target_mask] = 0.0
        del loaded, labels
        if mode == "label_isolated" and "target_labels" in child_payload:
            raise RuntimeError("isolated child payload has labels")

        context = get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=optimizer_child_entry,
            args=(child_payload, sender),
            name=f"parity-{dataset}-s{session}-p{subject}-{mode}",
        )
        process.daemon = True
        try:
            process.start()
            sender.close()
            del child_payload
            process.join()
            message = receiver.recv() if receiver.poll() else None
        finally:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5.0)
            receiver.close()
            try:
                sender.close()
            except OSError:
                pass
        if process.exitcode != 0 or not message or message.get("ok") is not True:
            detail = message.get("traceback", message.get("error")) if message else "no child receipt"
            raise RuntimeError(
                f"spawned optimizer failed with exit code {process.exitcode}: {detail}"
            )

        optimization_files = set(EXPECTED_FILES[:4])
        actual_files = {entry.name for entry in fold_dir.iterdir()}
        if actual_files != optimization_files:
            raise RuntimeError(
                f"spawned optimizer artifact boundary mismatch: "
                f"expected={sorted(optimization_files)}, actual={sorted(actual_files)}"
            )
        probability_path = fold_dir / "probabilities.npz"
        log_path = fold_dir / "training_log.csv"
        manifest_path = fold_dir / "fold_manifest.json"
        receipt_path = fold_dir / "optimization_receipt.json"
        labels_path = fold_dir / "target_labels.npz"
        evaluation_path = fold_dir / "evaluation.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        probability_sha = sha256_file(probability_path)
        if receipt.get("probability_sha256") != probability_sha:
            raise RuntimeError("spawned optimizer probability receipt mismatch")
        if receipt.get("training_log_sha256") != sha256_file(log_path):
            raise RuntimeError("spawned optimizer log receipt mismatch")
        if receipt.get("fold_manifest_sha256") != sha256_file(manifest_path):
            raise RuntimeError("spawned optimizer manifest receipt mismatch")
        if receipt.get("target_labels_still_sealed") is not (mode == "label_isolated"):
            raise RuntimeError("spawned optimizer vault receipt mismatch")
        if manifest.get("code_sha256") != launch_code or manifest.get("git") != launch_git:
            raise RuntimeError("spawned optimizer manifest launch provenance mismatch")
        for document_name, document in (("manifest", manifest), ("receipt", receipt)):
            embedded_launch = document.get("launch_provenance", {})
            embedded_completion = document.get("completion_provenance", {})
            if (
                embedded_launch.get("code_sha256") != launch_code
                or embedded_launch.get("git") != launch_git
                or embedded_completion.get("code_sha256") != launch_code
                or embedded_completion.get("git") != launch_git
            ):
                raise RuntimeError(
                    f"spawned optimizer {document_name} phase provenance mismatch"
                )
        child_execution = manifest.get("execution_contract", {})
        if child_execution.get("optimizer_process_boundary") != "multiprocessing_spawn":
            raise RuntimeError("spawned optimizer boundary is absent from manifest")
        if child_execution.get("optimizer_child_received_target_labels") is not (
            mode == "upstream_target_assisted"
        ):
            raise RuntimeError("spawned optimizer manifest target-label receipt mismatch")
        if receipt.get("optimizer_process_boundary") != "multiprocessing_spawn":
            raise RuntimeError("spawned optimizer boundary is absent from receipt")
        if receipt.get("optimizer_child_pid") != process.pid:
            raise RuntimeError("spawned optimizer PID receipt mismatch")
        for key, value in canonical_label_flags(mode).items():
            if receipt.get(key) is not value or child_execution.get(key) is not value:
                raise RuntimeError(f"spawned optimizer canonical flag mismatch: {key}")

        parent_completion_provenance = verify_current_provenance(
            launch_provenance,
            context="in parent after optimizer artifacts before target labels open",
        )
        if mode == "label_isolated":
            target_one_hot = vault.open(
                "posthoc_evaluation", probability_sha256=probability_sha
            )
        else:
            target_one_hot = vault.open("target_assisted_optimization")

        with np.load(probability_path, allow_pickle=False) as probabilities:
            retained_mask = probabilities["retained_mask"].astype(np.bool_)
            retained_index = probabilities["retained_target_local_index"].astype(np.int32)
            retained_last_batched_probs = probabilities["retained_last_batched_probs"].copy()
            all_last_batched_probs = probabilities["all_last_batched_probs"].copy()
            retained_best_batched_probs = probabilities["retained_best_batched_probs"].copy()
            all_best_batched_probs = probabilities["all_best_batched_probs"].copy()
            trials = probabilities["all_trial"].astype(np.int16)
            sample_within_trial = probabilities["all_sample_within_trial"].astype(np.int32)
            original_index = probabilities["all_original_dataset_index"].astype(np.int32)
        if len(retained_mask) != len(target_data) or not np.array_equal(
            original_index, target_original_index
        ):
            raise RuntimeError("probability artifact target identity mismatch")

        all_y_true = target_one_hot.argmax(axis=1).astype(np.int16)
        retained_y_true = all_y_true[retained_mask]
        atomic_npz(
            labels_path,
            all_y_true=all_y_true,
            retained_y_true=retained_y_true,
            all_subject=np.full(len(target_data), subject, dtype=np.int16),
            all_session=np.full(len(target_data), session, dtype=np.int8),
            all_trial=trials,
            all_sample_within_trial=sample_within_trial,
            all_original_dataset_index=original_index,
            retained_target_local_index=retained_index,
        )

        retained_last_metrics = multiclass_metrics(
            retained_y_true, retained_last_batched_probs
        )
        all_last_metrics = multiclass_metrics(all_y_true, all_last_batched_probs)
        training_summary = receipt["training_summary"]
        if mode == "upstream_target_assisted":
            retained_best_metrics = multiclass_metrics(
                retained_y_true, retained_best_batched_probs
            )
            all_best_metrics = multiclass_metrics(all_y_true, all_best_batched_probs)
            reported_accuracy_percent = 100.0 * retained_best_metrics["accuracy"]
            if abs(
                reported_accuracy_percent
                - float(training_summary["best_target_accuracy_percent"])
            ) > 1e-6:
                raise RuntimeError(
                    "reloaded best state does not reproduce retained target accuracy"
                )
        else:
            retained_best_metrics = None
            all_best_metrics = None

        evaluation = {
            "study": STUDY,
            "protocol_version": PROTOCOL_VERSION,
            "run_tag": run_tag,
            "mode": mode,
            "dataset": dataset,
            "session": session,
            "target_subject": subject,
            "source_subjects": source_subjects,
            "source_subject_count": 14,
            "seed": seed,
            "max_iter": max_iter,
            "iterations_completed": training_summary["iterations_completed"],
            "fixed_final_available": training_summary["iterations_completed"] == max_iter,
            "best_iteration": training_summary["best_iteration"],
            "best_target_accuracy_percent": training_summary["best_target_accuracy_percent"],
            "stopping_reason": training_summary["stopping_reason"],
            "source_iterator_resets": training_summary["source_iterator_resets"],
            "target_iterator_resets": training_summary["target_iterator_resets"],
            "training_seconds": float(training_summary["training_seconds"]),
            "retained_target_samples": int(retained_mask.sum()),
            "all_target_samples": int(len(target_data)),
            "retained_last_at_stop": retained_last_metrics,
            "all_last_at_stop": all_last_metrics,
            "retained_target_best": retained_best_metrics,
            "all_target_best": all_best_metrics,
            "reported_endpoint": (
                "iteration_1000_fixed_final"
                if mode == "label_isolated"
                else "maximum_held_out_target_accuracy_earliest_tie"
            ),
            "target_labels_available_to_optimization": mode == "upstream_target_assisted",
            "target_labels_used_in_loss": False,
            "target_labels_used_for_checkpoint_selection": mode == "upstream_target_assisted",
            "target_labels_used_for_early_stopping": mode == "upstream_target_assisted",
            "target_labels_opened_after_probability_commit": mode == "label_isolated",
            "optimizer_process_boundary": "multiprocessing_spawn",
            "optimizer_child_received_target_labels": mode == "upstream_target_assisted",
            **canonical_label_flags(mode),
            "label_vault_final_state": vault.state,
            "label_vault_transitions": vault.transitions,
            "artifact_sha256": {
                "probabilities.npz": probability_sha,
                "training_log.csv": sha256_file(log_path),
                "fold_manifest.json": sha256_file(manifest_path),
                "optimization_receipt.json": sha256_file(receipt_path),
                "target_labels.npz": sha256_file(labels_path),
            },
            "target_labels_sha256": sha256_array(target_one_hot),
            "code_combined_sha256": launch_code["combined_sha256"],
            "git_revision": launch_git["revision"],
            "launch_provenance": launch_provenance,
            "completion_provenance": parent_completion_provenance,
        }
        atomic_json(evaluation_path, evaluation)
        if not validate_completed_fold(fold_dir, **validation_args):
            raise RuntimeError("completed fold failed final validation")
        print(json.dumps(evaluation, indent=2, sort_keys=True), flush=True)
        return fold_dir
    finally:
        live_optimizer = (
            process is not None
            and process.pid is not None
            and process.is_alive()
        )
        if live_optimizer:
            print(
                f"optimizer child is still alive; retaining lock claim: {claim_path}",
                file=sys.stderr,
                flush=True,
            )
        else:
            release_fold_claim(claim_path, claim_token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_CONTRACT), required=True)
    parser.add_argument("--session", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--subject", type=int, choices=range(1, 16), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--run_tag", default=None)
    cli = parser.parse_args()
    run_tag = cli.run_tag or PAPER_TAGS[cli.mode]
    run_fold(
        dataset=cli.dataset,
        session=cli.session,
        subject=cli.subject,
        mode=cli.mode,
        seed=cli.seed,
        max_iter=cli.max_iter,
        run_tag=run_tag,
    )


if __name__ == "__main__":
    main()
