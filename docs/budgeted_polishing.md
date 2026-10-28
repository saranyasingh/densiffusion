# Prospective resource-matched history polishing

This experiment tests whether the earlier noisy-history polishing improvement
survives a prospectively matched full-pipeline **dense matrix work** budget.
It compares raw, learned blind-spot, and seven-point median histories on the
original noisy-periodic distribution. It performs no densification and cannot
establish resolution-transfer performance or an optimal allocation curve.

The scientific [configuration](../configs/history_polishing_budget_matched.json)
uses three new paired seeds and a new RNG namespace for every split: 256 train,
256 validation, and 2,048 test trajectories per seed; 32 integer-time historical
observations and eight unchanged noisy future targets. Clean signals and latent
parameters never enter any fitting, preprocessing, or selection API.

All forecasters use the recovered GP architecture: four residual blocks, 64
hidden channels, eight heads, 128-dimensional time/diffusion embeddings,
feature embedding 16, feedforward width 64, and dropout 0.1. They have fresh,
paired initial weights and share example-order and training-noise seeds. Adam,
quadratic 50-step diffusion, weight decay, gradient clipping, and learning-rate
decay are retained. LR milestones occur at 75% and 90% of each method's own
allotted training budget. Method order rotates across seeds.

The ceiling is **173,320,595,046,400 counted matrix operations per method per
seed**, including complete auxiliary search, history preprocessing, forecaster
optimization, all-step noise validation, and forecast-CRPS validation. One
multiply plus one add counts as two. Actual forward and gradient-hook counts
are checked against the plan. Independent formulas verify the profiler,
including temporal attention products and omitted unnecessary input gradients.

The complete four-candidate mediator search is charged: two widths, two losses,
200 epochs each, and masked noisy-center validation. Mediator selection uses
only reconstruction MSE. Every selected mediator is frozen before any
forecaster is trained. Raw and median forecasters receive 1,200 epochs / 4,800
updates. Learned polishing receives the maximal feasible whole-epoch budget,
1,192 epochs / 4,768 updates. Unspent counted work is below 0.01% of the common
ceiling, rather than the large residual gaps in the retrospective audit.

All methods receive seven forecast validation opportunities: epochs 10, 50,
100, 200, 400, 800, and their own final epoch. Each scores all 256 validation
trajectories with 64 draws. All-step noise validation occurs every ten epochs
and at the final epoch, giving 120 passes per method. Only original-grid
forecast CRPS selects a forecaster checkpoint. Auxiliary search is part of the
tested pipeline and charged in full; no additional hyperparameter search is
performed for any method. This is a comparison of these fixed pipelines, not
proof that every baseline has been globally optimized.

This matching unit excludes nonmatrix operations, Adam arithmetic, median
comparisons, memory traffic, and I/O. It is not an equal-energy or equal-wall-time
claim. Actual full elapsed times and deployment costs are recorded separately.
The data-free sampling-batch pilot selected 1,024 equally for all methods; it
used discarded random weights and zero histories, without quality scores.

All configurations, allocation plans, and selected models freeze before fresh
test data generation. The runner verifies training artifacts before beginning
the test phase. The evaluator refuses an existing test-evaluation marker.
Exact scored ensembles, four forecast metrics, per-horizon scores, 10,000 paired
trajectory-bootstrap replicates, and descriptive seed variation are saved by
the established analysis code. Test targets remain noisy. Bootstrap intervals
condition on fitted models and saved draws and are unadjusted for multiplicity.

```sh
uv run --locked python scripts/train_budgeted_polishing.py --smoke \
  --output outputs/budgeted-polishing-smoke
uv run --locked python scripts/train_budgeted_polishing.py \
  --config configs/history_polishing_budget_matched.json \
  --output outputs/budgeted-polishing-primary
uv run --locked python scripts/verify_budgeted_polishing.py \
  outputs/budgeted-polishing-primary
```

Use fresh destinations. The smoke run establishes integration only. The
scientific run uses one MPS worker and one Torch thread; it does not treat
completing the allotted epochs as evidence of convergence.
