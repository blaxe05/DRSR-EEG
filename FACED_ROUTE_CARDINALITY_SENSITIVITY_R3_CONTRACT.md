# FACED routing-cardinality sensitivity protocol

This experiment evaluates the frozen FACED DRSR model and compares
the original top-14 structural route with an all-122 source route at the same trained
state. It tests fixed-model prediction sensitivity to route cardinality without
changing training, the residual coefficient, source evidence, or the iteration-1000
endpoint.

## Evaluation protocol

The experiment evaluates participant-wise leave-one-subject-out folds across all 123
FACED participants. For each fold, predictions are evaluated between:
1. The primary sparse top-14 structural route.
2. The complete all-122 source aggregation route using normalized positive reliability weights.

Each fold evaluates classifier probabilities, top-14 route probabilities,
all-source route probabilities, source identities, and final reliability weights
with target labels withheld during training. Analysis computes participant-paired
accuracy, UAR, macro-F1, ECE, CVaR20 UAR, and P10 UAR using a 100,000-draw participant
bootstrap.
