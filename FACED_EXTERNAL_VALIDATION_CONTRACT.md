# FACED external-validation contract

## Scientific role

FACED is a frozen external validation of DRSR. SEED and SEED-IV remain the
development and matched-comparison datasets. FACED outcomes must not be used to
change the model, candidate lattice, source-routing rule, optimizer, checkpoint
schedule, preprocessing choice, or statistical test.

## Input contract

- Dataset root: supplied locally through the `FACED_DATA_ROOT` environment variable
- Model input: official differential-entropy tensors in `EEG_Features\DE`
- Expected participant lattice: 123 files, `sub000.pkl.pkl` through
  `sub122.pkl.pkl`
- Expected tensor shape: 28 videos x 32 channels x 30 one-second windows x
  5 frequency bands
- Class mapping: anger, disgust, fear, sadness, neutral, amusement,
  inspiration, joy, tenderness in the official stimulus order
- Derived representation: deterministic per-video LDS followed by
  participant-wise feature scaling to [-1, 1]
- No official source file is modified

## Evaluation contract

- Held-out participant labels are unavailable to normalization, clustering,
  optimization, routing, model selection, and fixed-final checkpoint selection.
- Fixed-final iteration 1000 is the primary endpoint.
- A target-assisted maximum over the already declared 50-step checkpoint grid
  may be reported only as a separate diagnostic.
- UAR and macro-F1 are primary. Accuracy, calibration, class-wise recall, and
  participant-level lower-tail summaries are secondary.
- Both residual inputs come from the same trained DRSR model. The
  discriminative term is `softmax(g_phi(f_theta(x)))`, and the structural term
  is reconstructed from the selected source routes. No separately trained
  parent probability is used.
- Source and target representation banks use corrected cluster means. Their
  history-retention coefficients are 0.5 and 0.1, respectively, matching the
  method specification. Structural inference uses two-hop nearest-cluster
  voting followed by the frozen sparse source route.
- Inference and uncertainty summaries are participant based. FACED is a frozen
  external validation of the final DRSR model, not a parent-model comparison.

## Feasibility decision

The smoke profiles both 14-source and 122-source configurations using the same
participant target, batch size, active model path, and number of timed steps.
Performance is not evaluated. The full-source protocol is retained only when
the measured memory and projected runtime are operationally feasible. Any
resource fallback must use a deterministic 14-source panel declared before
outcomes are computed.

## Execution safeguards

Validation checks for missing or malformed tensors, non-finite values, and target-label exposure.
The residual classifier probabilities are generated directly by the active DRSR model
to ensure identical model representation across all evaluated folds.
