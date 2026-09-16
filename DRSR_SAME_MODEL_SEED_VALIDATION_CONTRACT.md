# Corrected same-model SEED and SEED-IV validation contract

## Scientific role

This run re-estimates the final DRSR configuration on SEED and SEED-IV after
the same-model residual definition and corrected cluster-centre implementation
were frozen for FACED. It does not compare against, import from, or mix
probabilities from HEDN.

## Evaluation contract

- Each dataset is evaluated with session-specific leave-one-participant-out
  validation, giving 45 outer folds per dataset and 90 folds in total.
- Held-out labels are unavailable to normalization, clustering, training,
  routing, or fixed-final selection.
- Iteration 1000 is the primary label-isolated endpoint.
- Target-assisted selection over the prespecified checkpoints 50, 100, ...,
  1000 is a separate, nondeployable checkpoint-sensitivity diagnostic.
- Both residual inputs come from the same trained DRSR model. The
  discriminative probability is produced by its classifier head and the
  structural probability by its selected source routes.
- The frozen residual coefficient is 0.25, routing cardinality is 14,
  routing temperature is 1.0, and both JS and conditional-alignment
  coefficients are zero. Source-only warm-up is disabled.
- Source and target representation banks use corrected cluster means.
- Optimizer seed 42, preprocessing, checkpoint lattice, and model parameters
  are identical across all folds except for the dataset-dependent class count
  and batch size.

## Execution protocol

- Folds are executed across SEED and SEED-IV datasets with up to two concurrent workers.
- The pipeline validates tensor shapes, label isolation, and numerical reconstruction before committing fold results.

