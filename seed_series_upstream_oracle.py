"""One SEED/SEED-IV fold under the original HEDN target-best protocol.

HEDN is the upstream reproduction arm. RW-HEDN and Proposed are our models
transposed onto the same data, clustering, optimization, and checkpoint rule.
All outputs are explicitly target-label-assisted oracle diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from config import get_parser
from data_utils._get_dataset import get_dataset
from data_utils.HEDNLoader import HEDNLoader
from get_model_utils import get_model_utils
from utils.utils import setup_seed


HERE = Path(__file__).resolve().parent
RUN_TAG = "paper_seed_series_upstream_oracle_v1"
STUDY = "seed_series_upstream_oracle"
UPSTREAM_COMMIT = "529e417a258a687c8d0239496b2b01d52cbed644"
MODEL_MAP = {"hedn": "hedn", "rwhedn": "rwhedn", "proposed": "asjda_hedn_v2"}
DATASET_CONTRACT = {
    "seed3": {
        "label": "SEED", "num_classes": 3, "batch_size": 96,
        "samples_per_subject": {1: 3394, 2: 3394, 3: 3394},
    },
    "seed4": {
        "label": "SEED-IV", "num_classes": 4, "batch_size": 64,
        "samples_per_subject": {1: 851, 2: 832, 3: 822},
    },
}
PROPOSED_LOCK = {
    "cond_align_weight": 0.02,
    "warmup_iters": 300,
    "pseudo_conf_quantile": 0.75,
    "pseudo_conf_quantile_start": 0.85,
    "pseudo_conf_quantile_end": 0.65,
    "pseudo_conf_quantile_iters": 400,
    "js_weight": 1.0,
    "rel_momentum": 0.95,
    "weight_decay": 5e-5,
    "label_smoothing": 0.05,
    "feature_dropout": 0.10,
    "feature_noise_std": 0.02,
    "source_only_warmup_iters": 200,
    "adapt_ramp_iters": 200,
    "cond_ramp_iters": 300,
    "cons_ramp_iters": 200,
    "classwise_confidence": True,
    "min_tgt_class_mass": 0.50,
    "min_tgt_class_count": 4,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def write_json_immutable(path: Path, payload: dict) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"immutable JSON differs: {path}")
        return
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def write_npz_immutable(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays):
                raise RuntimeError(f"immutable NPZ keys differ: {path}")
            for key, value in arrays.items():
                if not np.array_equal(np.asarray(old[key]), np.asarray(value)):
                    raise RuntimeError(f"immutable NPZ differs: {path}:{key}")
        return
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def git_provenance() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(args, cwd=HERE, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unavailable"
    status = run("git", "status", "--short")
    return {
        "revision": run("git", "rev-parse", "HEAD"),
        "dirty": bool(status and status != "unavailable"),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def build_args(dataset: str, session: int, model: str, seed: int, max_iter: int):
    spec = DATASET_CONTRACT[dataset]
    args = get_parser().parse_args([
        "--dataset_name", dataset,
        "--session", str(session),
        "--model_name", MODEL_MAP[model],
        "--rw_stage", "full",
        "--seed", str(seed),
        "--max_iter", str(max_iter),
        "--early_stop", str(max_iter),
        "--log_interval", "50",
        "--batch_size", str(spec["batch_size"]),
        "--num_subjects", "15",
        "--num_sources", "14",
        "--num_workers", "0",
        "--balance_source_classes", "false",
    ])
    args.device = torch.device("cuda:0")
    args.num_classes = spec["num_classes"]
    args.feature_dim = 310
    args.weight_decay = 1e-5
    if model == "proposed":
        for key, value in PROPOSED_LOCK.items():
            setattr(args, key, value)
    return args


def metric_bundle(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> dict:
    pred = probs.argmax(1).astype(np.int16)
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "uar": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "classwise_f1": f1_score(y_true, pred, labels=labels, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).astype(int).tolist(),
    }


@torch.no_grad()
def predict_batched(model, data: torch.Tensor | np.ndarray, batch_size: int = 2048) -> np.ndarray:
    tensor = data if isinstance(data, torch.Tensor) else torch.from_numpy(np.asarray(data)).float()
    outputs = []
    model.eval()
    for start in range(0, len(tensor), batch_size):
        outputs.append(model.predict_proba(tensor[start:start + batch_size].to("cuda:0"), mode="target").cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def within_trial_index(trials: np.ndarray) -> np.ndarray:
    result = np.empty(len(trials), dtype=np.int32)
    for trial in np.unique(trials):
        mask = trials == trial
        result[mask] = np.arange(int(mask.sum()), dtype=np.int32)
    return result


def validate_dataset(dataset: str, session: int, data: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> None:
    spec = DATASET_CONTRACT[dataset]
    expected_per_subject = spec["samples_per_subject"][session]
    expected_total = 15 * expected_per_subject
    if data.shape != (expected_total, 310):
        raise RuntimeError(f"data shape mismatch: {data.shape}")
    if labels.shape != (expected_total, spec["num_classes"]):
        raise RuntimeError(f"label shape mismatch: {labels.shape}")
    if groups.shape != (expected_total, 3):
        raise RuntimeError(f"group shape mismatch: {groups.shape}")
    if set(np.unique(groups[:, 0]).tolist()) != set(range(1, 16)):
        raise RuntimeError("subject identity mismatch")
    if set(np.unique(groups[:, 2]).tolist()) != {session}:
        raise RuntimeError("session identity mismatch")
    counts = [int(np.sum(groups[:, 0] == subject)) for subject in range(1, 16)]
    if counts != [expected_per_subject] * 15:
        raise RuntimeError(f"per-subject count mismatch: {counts}")
    if not np.isfinite(data).all() or float(data.min()) < -1.000001 or float(data.max()) > 1.000001:
        raise RuntimeError("feature scaling/finite-value contract failed")
    if not np.allclose(labels.sum(1), 1.0):
        raise RuntimeError("labels are not one-hot")


def run_fold(dataset: str, session: int, subject: int, model: str, seed: int, max_iter: int, run_tag: str) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if dataset not in DATASET_CONTRACT or model not in MODEL_MAP:
        raise ValueError((dataset, model))
    if session not in (1, 2, 3) or subject not in range(1, 16):
        raise ValueError((session, subject))

    setup_seed(seed)
    args = build_args(dataset, session, model, seed, max_iter)
    loaded = get_dataset(args)
    data = np.asarray(loaded["data"])
    labels = np.asarray(loaded["labels"], dtype=np.float32)
    groups = np.asarray(loaded["groups"], dtype=np.int16)
    validate_dataset(dataset, session, data, labels, groups)

    target_mask = groups[:, 0] == subject
    source_mask = ~target_mask
    source_subjects = sorted(np.unique(groups[source_mask, 0]).astype(int).tolist())
    if len(source_subjects) != 14 or subject in source_subjects:
        raise RuntimeError("LOSO source identity contract failed")
    train_dataset = {"data": data[source_mask], "labels": labels[source_mask], "groups": groups[source_mask]}
    target_dataset = {"data": data[target_mask], "labels": labels[target_mask], "groups": groups[target_mask]}

    loader_builder = HEDNLoader(args, train_dataset, target_dataset)
    train_loader, target_loader = loader_builder()
    cluster_params = {
        "eps": float(loader_builder.best_cluster_params["eps"]),
        "min_samples": int(loader_builder.best_cluster_params["min_samples"]),
    }
    retained_mask = DBSCAN(**cluster_params).fit(target_dataset["data"]).labels_ != -1
    if int(retained_mask.sum()) != len(target_loader.dataset):
        raise RuntimeError("target retained-mask mismatch")

    trainer = get_model_utils(args)
    started = time.time()
    oracle_best_accuracy, training_log = trainer.train(train_loader, target_loader)
    elapsed = time.time() - started
    retained_x, retained_labels, _ = target_loader.dataset.get_data()
    y_true = retained_labels.argmax(1).numpy().astype(np.int16)
    last_probs = predict_batched(trainer.model, retained_x)
    all_last_probs = predict_batched(trainer.model, target_dataset["data"])
    last_metrics = metric_bundle(y_true, last_probs, args.num_classes)
    if trainer.get_best_model_state() is None:
        raise RuntimeError("target-best model state was not created")
    trainer.model.load_state(trainer.get_best_model_state())
    best_probs = predict_batched(trainer.model, retained_x)
    all_best_probs = predict_batched(trainer.model, target_dataset["data"])
    best_metrics = metric_bundle(y_true, best_probs, args.num_classes)
    if abs(best_metrics["accuracy"] * 100.0 - float(oracle_best_accuracy)) > 1e-6:
        raise RuntimeError("reloaded best checkpoint does not reproduce reported accuracy")

    target_groups = groups[target_mask]
    all_y_true = labels[target_mask].argmax(1).astype(np.int16)
    trials = target_groups[:, 1].astype(np.int16)
    sample_within_trial = within_trial_index(trials)
    original_index = np.flatnonzero(target_mask).astype(np.int32)
    retained_index = np.flatnonzero(retained_mask).astype(np.int32)

    fold_dir = HERE / "results" / STUDY / run_tag / dataset / f"session_{session}" / model / f"target_subject_{subject:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = fold_dir / "oracle_predictions.npz"
    log_path = fold_dir / "training_log.csv"
    manifest_path = fold_dir / "fold_manifest.json"
    evaluation_path = fold_dir / "evaluation.json"
    paths = (prediction_path, log_path, manifest_path, evaluation_path)
    if any(path.exists() for path in paths) and not all(path.exists() for path in paths):
        raise RuntimeError(f"partial fold artifacts exist: {fold_dir}")

    write_npz_immutable(
        prediction_path,
        subject=np.full(len(y_true), subject, dtype=np.int16),
        session=np.full(len(y_true), session, dtype=np.int8),
        trial=trials[retained_mask],
        sample_within_trial=sample_within_trial[retained_mask],
        original_dataset_index=original_index[retained_mask],
        target_local_index=retained_index,
        y_true=y_true,
        best_pred=best_probs.argmax(1).astype(np.int16),
        last_pred=last_probs.argmax(1).astype(np.int16),
        best_probs=best_probs,
        last_probs=last_probs,
        all_trial=trials,
        all_sample_within_trial=sample_within_trial,
        all_original_dataset_index=original_index,
        all_y_true=all_y_true,
        all_best_probs=all_best_probs,
        all_last_probs=all_last_probs,
        retained_mask=retained_mask.astype(np.bool_),
    )
    if log_path.exists():
        old = np.atleast_2d(np.loadtxt(log_path, delimiter=",", skiprows=1))
        if old.shape != training_log.shape or not np.allclose(old, training_log, atol=5e-7):
            raise RuntimeError(f"immutable training log differs: {log_path}")
    else:
        header = "cls_loss,transfer_loss,cons_loss,src_cluster_loss,tgt_cluster_loss,source_accuracy,target_accuracy,best_target_accuracy,easy_source,hard_source"
        tmp = log_path.with_suffix(f".csv.{os.getpid()}.tmp")
        np.savetxt(tmp, training_log, delimiter=",", header=header, comments="", fmt="%.8f")
        os.replace(tmp, log_path)

    source_path = args.seed3_path if dataset == "seed3" else args.seed4_path
    model_config = {
        "model_name": args.model_name,
        "rw_stage": args.rw_stage,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "transfer_loss_weight": args.transfer_loss_weight,
        "constraint_loss_weight": args.constraint_loss_weight,
        "src_momentum": args.src_momentum,
        "tgt_momentum": args.tgt_momentum,
    }
    if model == "proposed":
        model_config.update(PROPOSED_LOCK)
    manifest = {
        "study": STUDY,
        "protocol_version": "v1",
        "run_tag": run_tag,
        "dataset": dataset,
        "dataset_label": DATASET_CONTRACT[dataset]["label"],
        "session": session,
        "model": model,
        "model_role": "upstream reproduction" if model == "hedn" else "protocol-transposed authors' model",
        "selection_reference": "asjda_hedn_v2_l1l2_full" if model == "proposed" else ("rwhedn_full" if model == "rwhedn" else "upstream_hedn"),
        "target_subject": subject,
        "source_subjects": source_subjects,
        "source_subject_count": 14,
        "seed": seed,
        "max_iter": max_iter,
        "upstream_commit": UPSTREAM_COMMIT,
        "model_config": model_config,
        "protocol": {
            "session_specific_loso": True,
            "batch_size": args.batch_size,
            "feature_dim": 310,
            "num_classes": args.num_classes,
            "normalization": "upstream participant-wise min-max to [-1,1] within session",
            "target_labels_used_for_checkpoint_selection": True,
            "target_label_isolated": False,
            "diagnostic_only": True,
            "checkpoint_rule": "maximum held-out-target retained-window accuracy over all iterations",
            "reporting_grain": "DBSCAN-retained precomputed DE windows",
            "primary_historical_metric": "accuracy",
            "target_noise_windows_excluded_from_historical_metric": True,
            "dbscan_selected_from_first_source_subject_labels": True,
            "dbscan_params": cluster_params,
        },
        "target_coverage": {
            "expected_windows": int(target_mask.sum()),
            "retained_windows": int(retained_mask.sum()),
            "fraction": float(retained_mask.mean()),
        },
        "input_provenance": {
            "configured_source_path": str(Path(source_path).resolve()),
            "data_sha256": sha256_array(data),
            "labels_sha256": sha256_array(labels),
            "groups_sha256": sha256_array(groups),
            "data_shape": list(data.shape),
            "data_dtype": str(data.dtype),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "git": git_provenance(),
        "code_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "get_model_utils.py": sha256_file(HERE / "get_model_utils.py"),
            "HEDNTrainer.py": sha256_file(HERE / "trainers" / "HEDNTrainer.py"),
            "HEDNLoader.py": sha256_file(HERE / "data_utils" / "HEDNLoader.py"),
            "_get_dataset.py": sha256_file(HERE / "data_utils" / "_get_dataset.py"),
            "model_file": sha256_file(HERE / "models" / ({"hedn": "HEDN.py", "rwhedn": "RWHEDN.py", "proposed": "ASJDAHEDN.py"}[model])),
        },
    }
    write_json_immutable(manifest_path, manifest)
    evaluation = {
        "study": STUDY,
        "protocol_version": "v1",
        "run_tag": run_tag,
        "dataset": dataset,
        "session": session,
        "model": model,
        "target_subject": subject,
        "seed": seed,
        "oracle_target_best_accuracy_percent": float(oracle_best_accuracy),
        "best_metrics": best_metrics,
        "last_metrics": last_metrics,
        "all_target_best_metrics": metric_bundle(all_y_true, all_best_probs, args.num_classes),
        "all_target_last_metrics": metric_bundle(all_y_true, all_last_probs, args.num_classes),
        "best_iteration": int(np.argmax(training_log[:, 7]) + 1),
        "iterations_completed": int(len(training_log)),
        "training_seconds": float(elapsed),
        "target_coverage": manifest["target_coverage"],
        "target_labels_used_for_checkpoint_selection": True,
        "target_label_isolated": False,
        "diagnostic_only": True,
        "prediction_sha256": sha256_file(prediction_path),
        "training_log_sha256": sha256_file(log_path),
        "fold_manifest_sha256": sha256_file(manifest_path),
    }
    write_json_immutable(evaluation_path, evaluation)
    print(json.dumps(evaluation, indent=2, sort_keys=True))
    return fold_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_CONTRACT), required=True)
    parser.add_argument("--session", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--subject", type=int, choices=range(1, 16), required=True)
    parser.add_argument("--model", choices=sorted(MODEL_MAP), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--run_tag", default=RUN_TAG)
    cli = parser.parse_args()
    run_fold(cli.dataset, cli.session, cli.subject, cli.model, cli.seed, cli.max_iter, cli.run_tag)


if __name__ == "__main__":
    main()
