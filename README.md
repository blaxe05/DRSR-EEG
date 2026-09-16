# DRSR for cross-subject EEG emotion recognition

This repository contains the reproduction code for **Learning Complementary Source Roles for Cross-Subject EEG Emotion Recognition**. It provides the DRSR model architectures, dataset loaders, frozen experiment specifications, and experiment runners to reproduce the empirical results.

## What is included

- `models/`: DRSR source-routing and ablation modules (`DRSR.py`, `RWHEDN.py`, `RWHEDNJS.py`, `RWHEDNPseudoConditional.py`, `RWHEDNRoutingAblation.py`).
- `datasets/faced_feature.py`: Loader for the released FACED differential-entropy features.
- Experiment reproduction programs: `core_routing_ablation.py`, `drsr_same_model_seed_validation.py`, `faced_external_validation.py`, `faced_route_cardinality_sensitivity_r3.py`, `seed_series_upstream_oracle.py`, and supporting stage scripts (`upstream_v2_*.py`).
- Root design JSON files and Markdown contracts: Frozen experiment definitions (`*_design.json`, `*_CONTRACT.md`) specifying exact candidate grids, checkpoint cadences, and evaluation protocols.
- `verify_release.py`: Static integrity checks verifying frozen configurations and required reproduction assets.

Raw EEG, licensed feature files, model checkpoints, and per-window probability artifacts are not redistributed.

## Upstream runtime

The operational experiments were implemented as extensions to the public HEDN research code at commit `44b79852fdcbed7264ef6b14452b6a9e17ba48e0`:

```text
https://github.com/qwangwl/HEDN
```

The upstream repository did not contain a license file when this artifact was prepared, so its source is not duplicated here. HEDN is an implementation dependency only and is not a comparator or evidential parent of the article. To reproduce the training environment, check out that commit and overlay the files from this repository at the same relative paths. The DRSR modules import the upstream feature extractor, trainer, data utilities, and losses.

## Environment

The validated runs used Python 3.10, PyTorch with CUDA, NumPy, SciPy, scikit-learn, pandas, matplotlib, and PyYAML. Create an isolated environment and install the versions in `requirements.txt`. Dataset paths are supplied locally. For FACED, set:

```powershell
$env:FACED_DATA_ROOT = 'D:\path\to\FACED'
```

SEED and SEED-IV follow the directory contract of the upstream loader. The datasets must be obtained from their official custodians and are not included.

## Reproduction order

1. Obtain the official processed SEED, SEED-IV, and FACED features.
2. Clone and check out the upstream runtime commit shown above.
3. Overlay this repository onto that checkout.
4. Review the frozen JSON and Markdown contracts before launching a run.
5. Run the fold programs (e.g., `drsr_same_model_seed_validation.py`, `faced_external_validation.py`, `core_routing_ablation.py`).

Raw licensed EEG data, intermediate checkpoints, and machine-specific cache files are excluded from the repository. All model architectures, dataset loaders, configuration definitions, and reproduction scripts are included.

The target-assisted endpoint uses held-out labels after the complete trajectory has been saved. It is an oracle diagnostic and is not a deployable operating rule. The fixed-final endpoint is the primary matched comparison.

## Citation

Please cite the accompanying article. A machine-readable record is provided in `CITATION.cff`.

## Licensing and data rights

Copyright in the files authored for this study remains with the authors. No license is granted for the upstream HEDN source or for SEED, SEED-IV, or FACED. Users must comply with the terms imposed by the respective code and dataset custodians.
