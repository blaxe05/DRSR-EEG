# Core Routing Ablation Contract

## Objective

Test whether DRSR's complementary structural and corrective source routes are
supported under controlled component removal.  This study changes only the
source-weight assignment.  It does not reselect data, features, training
hyperparameters, checkpoint policy, top-k, residual gate, warm-up, normalized
JS term, or optional conditional term.

## Parent evidence

The immutable parent is Stage E2 production tag `e2r12p_20260820`.  For every
dataset, session, and target participant, the ablation imports the exact
source-only selections recorded by that fold.  The outer target labels remain
sealed across optimizer construction and training and are opened only after
the probability artifact is committed.

## Prespecified variants

- `uniform`: uniform structural and corrective weights.
- `shared_score`: the structural score distribution is shared by both roles.
- `structural_only`: structural weights are retained and corrective weights
  are uniform.
- `corrective_only`: corrective weights are retained and structural weights
  are uniform.
- `full`: an executable smoke-only equivalence control that delegates to the
  immutable parent implementation.

The primary production comparison imports `full` from the sealed parent and
runs the four component-removal variants on all 90 folds.  UAR at the
label-isolated fixed-final checkpoint is primary.  Accuracy, macro-F1,
calibration, participant-paired effects, and session-level results are
secondary.

## Execution parameters

- Random seed 42.
- SEED and SEED-IV datasets across sessions 1 to 3 and participants 1 to 15 (90 outer folds).
- Target labels strictly withheld during model fitting and weight assignment.
- Metric calculation uses double-precision probabilities to ensure exact numerical agreement.
