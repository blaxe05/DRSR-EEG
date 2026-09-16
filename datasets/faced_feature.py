"""Leakage-aware loader for the official FACED differential-entropy features.

The released feature files contain one NumPy array per participant with shape
``(video, channel, window, band) == (28, 32, 30, 5)``.  This adapter exposes
the same ``data``/``labels``/``groups`` contract used by the SEED loaders while
keeping participant and video identities available for participant-wise
evaluation.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterable

import numpy as np


class FACEDFeatureDataset:
    """Load FACED DE features as 1-second EEG windows.

    Parameters
    ----------
    root_path:
        FACED release root containing ``EEG_Features/DE``.
    subjects:
        Optional one-based participant identifiers in ``1..123``.
    apply_lds:
        Apply the same deterministic per-video linear dynamical smoothing used
        by the repository's DEAP adapter.  It uses features only and therefore
        does not cross the held-out-label boundary.
    scale_per_subject:
        Min-max scale every feature within each participant to ``[-1, 1]``,
        matching the participant-wise scaling used for SEED and SEED-IV.
    """

    NUM_SUBJECTS = 123
    NUM_VIDEOS = 28
    NUM_CHANNELS = 32
    NUM_WINDOWS = 30
    NUM_BANDS = 5
    NUM_CLASSES = 9
    FEATURE_DIM = NUM_CHANNELS * NUM_BANDS
    EXPECTED_SHAPE = (NUM_VIDEOS, NUM_CHANNELS, NUM_WINDOWS, NUM_BANDS)

    CLASS_NAMES = (
        "anger",
        "disgust",
        "fear",
        "sadness",
        "neutral",
        "amusement",
        "inspiration",
        "joy",
        "tenderness",
    )

    # Official stimulus order: 3 videos for each class except 4 neutral videos.
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

    _FILE_PATTERN = re.compile(r"^sub(?P<subject>\d{3})\.pkl\.pkl$")

    def __init__(
        self,
        root_path: str | Path,
        subjects: Iterable[int] | None = None,
        *,
        apply_lds: bool = True,
        scale_per_subject: bool = True,
    ) -> None:
        self.root_path = Path(root_path)
        self.feature_root = self.root_path / "EEG_Features" / "DE"
        self.subjects = self._validate_subjects(subjects)
        self.apply_lds = bool(apply_lds)
        self.scale_per_subject = bool(scale_per_subject)
        self._dataset_cache = self._load()

    def get_dataset(self) -> dict[str, np.ndarray]:
        return self._dataset_cache

    def get_feature_dim(self) -> int:
        return self.NUM_BANDS

    @classmethod
    def _validate_subjects(cls, subjects: Iterable[int] | None) -> tuple[int, ...]:
        selected = (
            tuple(range(1, cls.NUM_SUBJECTS + 1))
            if subjects is None
            else tuple(int(value) for value in subjects)
        )
        if not selected:
            raise ValueError("at least one FACED participant is required")
        if len(set(selected)) != len(selected):
            raise ValueError("FACED participant identifiers must be unique")
        invalid = [value for value in selected if value < 1 or value > cls.NUM_SUBJECTS]
        if invalid:
            raise ValueError(f"FACED participant identifiers outside 1..123: {invalid}")
        return tuple(sorted(selected))

    def _feature_path(self, subject: int) -> Path:
        path = self.feature_root / f"sub{subject - 1:03d}.pkl.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"missing FACED DE feature file: {path}")
        match = self._FILE_PATTERN.match(path.name)
        if match is None or int(match.group("subject")) != subject - 1:
            raise RuntimeError(f"FACED feature identity mismatch: {path}")
        return path

    @staticmethod
    def _load_numpy_pickle(path: Path) -> np.ndarray:
        # The release stores plain NumPy arrays.  Refuse object arrays and any
        # unexpected tensor contract immediately after deserialization.
        with path.open("rb") as handle:
            value = pickle.load(handle)
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise RuntimeError(f"object-valued FACED feature tensor: {path}")
        return array

    @staticmethod
    def _lds(data: np.ndarray) -> np.ndarray:
        """Apply the repository's scalar Kalman smoother along window time."""
        num_t, num_channel, num_feature = data.shape
        flat = np.asarray(data, dtype=np.float64).reshape(num_t, -1)
        prior_correlation = 0.01
        transition_matrix = 1.0
        noise_correlation = 0.0001
        observation_matrix = 1.0
        observation_correlation = 1.0
        mean = flat.mean(axis=0)
        observations = flat.T
        num_features, num_samples = observations.shape
        predicted = np.zeros_like(observations)
        estimate = np.zeros_like(observations)
        gain = np.zeros_like(observations)
        variance = np.zeros_like(observations)
        gain[:, 0] = (
            prior_correlation
            * observation_matrix
            / (
                observation_matrix
                * prior_correlation
                * observation_matrix
                + observation_correlation
            )
        )
        estimate[:, 0] = mean + gain[:, 0] * (
            observations[:, 0] - observation_matrix * prior_correlation
        )
        variance[:, 0] = (
            np.ones(num_features) - gain[:, 0] * observation_matrix
        ) * prior_correlation
        for index in range(1, num_samples):
            predicted[:, index - 1] = (
                transition_matrix
                * variance[:, index - 1]
                * transition_matrix
                + noise_correlation
            )
            gain[:, index] = (
                predicted[:, index - 1]
                * observation_matrix
                / (
                    observation_matrix
                    * predicted[:, index - 1]
                    * observation_matrix
                    + observation_correlation
                )
            )
            estimate[:, index] = (
                transition_matrix * estimate[:, index - 1]
                + gain[:, index]
                * (
                    observations[:, index]
                    - observation_matrix
                    * transition_matrix
                    * estimate[:, index - 1]
                )
            )
            variance[:, index] = (
                1.0 - gain[:, index] * observation_matrix
            ) * predicted[:, index - 1]
        return estimate.T.reshape(num_t, num_channel, num_feature)

    @staticmethod
    def _minmax_subject(data: np.ndarray) -> np.ndarray:
        minimum = data.min(axis=0)
        maximum = data.max(axis=0)
        span = maximum - minimum
        scaled = np.zeros_like(data, dtype=np.float64)
        nonconstant = span > 0.0
        scaled[:, nonconstant] = (
            2.0 * (data[:, nonconstant] - minimum[nonconstant]) / span[nonconstant]
            - 1.0
        )
        return scaled

    def _load_subject(self, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path = self._feature_path(subject)
        tensor = self._load_numpy_pickle(path)
        if tensor.shape != self.EXPECTED_SHAPE:
            raise RuntimeError(
                f"FACED feature shape mismatch for participant {subject}: "
                f"{tensor.shape} != {self.EXPECTED_SHAPE}"
            )
        if not np.isfinite(tensor).all():
            raise RuntimeError(f"non-finite FACED feature value: {path}")

        # video, channel, window, band -> video, window, channel, band
        videos = np.asarray(tensor, dtype=np.float64).transpose(0, 2, 1, 3)
        if self.apply_lds:
            videos = np.stack([self._lds(video) for video in videos], axis=0)
        data = videos.reshape(self.NUM_VIDEOS * self.NUM_WINDOWS, self.FEATURE_DIM)
        if self.scale_per_subject:
            data = self._minmax_subject(data)

        labels = np.repeat(self.VIDEO_LABELS, self.NUM_WINDOWS)
        trials = np.repeat(np.arange(1, self.NUM_VIDEOS + 1, dtype=np.int16), self.NUM_WINDOWS)
        groups = np.column_stack(
            (
                np.full(len(labels), subject, dtype=np.int16),
                trials,
                np.ones(len(labels), dtype=np.int16),
            )
        )
        return (
            np.ascontiguousarray(data, dtype=np.float32),
            np.eye(self.NUM_CLASSES, dtype=np.float32)[labels],
            np.ascontiguousarray(groups, dtype=np.int16),
        )

    def _load(self) -> dict[str, np.ndarray]:
        if not self.feature_root.is_dir():
            raise FileNotFoundError(f"FACED DE feature directory is missing: {self.feature_root}")
        data_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        group_parts: list[np.ndarray] = []
        for subject in self.subjects:
            data, labels, groups = self._load_subject(subject)
            data_parts.append(data)
            label_parts.append(labels)
            group_parts.append(groups)
        return {
            "data": np.concatenate(data_parts, axis=0),
            "labels": np.concatenate(label_parts, axis=0),
            "groups": np.concatenate(group_parts, axis=0),
        }

