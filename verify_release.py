"""Run static checks that do not require licensed EEG data or the upstream runtime."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    training_designs = (
        "drsr_same_model_seed_design.json",
        "faced_external_validation_design.json",
    )
    for name in training_designs:
        payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
        assert "frozen_configuration" in payload, name
        assert payload["frozen_configuration"]["training_iterations"] == 1000, name
        checkpoints = payload["frozen_configuration"]["checkpoint_iterations"]
        assert checkpoints == list(range(50, 1001, 50)), name

    sensitivity_name = "faced_route_cardinality_sensitivity_r3_design.json"
    sensitivity = json.loads((ROOT / sensitivity_name).read_text(encoding="utf-8"))
    assert sensitivity["training_iterations"] == 1000, sensitivity_name
    assert sensitivity["checkpoint_iterations"] == list(range(50, 1001, 50)), sensitivity_name
    assert sensitivity["target_labels_used_for_training_or_policy_selection"] is False
    assert sensitivity["scientific_role"] == "fixed-training-state external routing-cardinality sensitivity"

    reproduction_files = (
        "models/DRSR.py",
        "datasets/faced_feature.py",
        "core_routing_ablation.py",
        "drsr_same_model_seed_validation.py",
        "faced_external_validation.py",
        "faced_route_cardinality_sensitivity_r3.py",
    )
    for rel_path in reproduction_files:
        assert (ROOT / rel_path).is_file(), rel_path
    print("release metadata and reproduction file checks passed")


if __name__ == "__main__":
    main()
