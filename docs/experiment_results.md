# Saved experiment results

The repository includes the complete local experiment archive under `outputs/`:
configurations, checkpoints, recorded data splits, exact scored ensembles,
training and validation logs, metrics, timings, plots, source snapshots, and
verification records. Original output files are preserved byte-for-byte.

| Experiment | Saved report or artifacts | Recorded result |
|---|---|---|
| Paired waveform forecasting | [Primary report](../outputs/waveform-primary-20261027/REPORT.md) | Mean CRPS: direct 0.146844, linear 0.136779, learned 0.365331. Learned densification lost in every seed. |
| Larger waveform imputation | [Larger-model report](../outputs/waveform-large-20261027/REPORT.md) | Training-geometry imputation improved, but the frozen resolution-transfer check failed. Forecasting did not proceed. |
| Noisy-history polishing | [Original polishing report](../outputs/history-polishing-20261027/REPORT.md) | Mean CRPS: raw 0.285006, learned 0.274400, median 0.282228. This run matched forecaster updates rather than total computation. |
| Resource accounting | [Compute-allocation report](../outputs/compute-allocation-20261027/REPORT.md) | Retrospective accounting motivated a prospective comparison; retrospective checkpoints do not establish a resource-matched optimum. |
| Resource-matched polishing | [Prospective comparison](../outputs/history-polishing-budget-matched-20261027/REPORT.md) | Mean CRPS: raw 0.284813, learned 0.272510, median 0.275057. Learned-minus-raw intervals favored learned in all three seeds; learned-minus-median did so in one. Median had better average MSE and coverage. |

The prospective comparison charges the full mediator search, preprocessing,
forecaster optimization, and validation against a common matrix-work ceiling.
Learned polishing receives 4,768 forecaster updates versus 4,800 for each
baseline. The total difference is below 0.005%; actual elapsed times are reported
separately. This is one budget point for history polishing. It does not establish
an optimal allocation, convergence, or a densification benefit.

The archive also retains [the successful GP imputation recipe](../outputs/csdi-gp-better/)
and [the earlier GP forecasting comparison](../outputs/csdi-forecast-larger-23001/),
including its separate final analysis. Smoke runs and intermediate diagnostics
remain available with their original provenance.

Run reports, Git snapshots, and `no_commit_or_push` fields describe the state at
the time each experiment completed. They predate the later user-authorized
publication of this archive and have not been rewritten. Verification helpers
that compare the current checkout's HEAD or source tree against that historical
state are historical audits; model/data verification scripts and saved-sample
metric recomputation remain available separately.

For reproduction commands and the limits of each claim, use the experiment's
report and linked documentation. New runs require fresh output directories;
existing test evidence is retained rather than overwritten.
