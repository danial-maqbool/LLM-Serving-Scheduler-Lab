# Delivery verification — 9 September 2026

## Published state

The study is implemented in Python source modules, not a notebook or web application.
The repository contains the three scheduler families, three cost-model interfaces, workload
construction/replay, resource invariants, experiment runner, statistics, figures, and report.

The original seven-test starter is preserved in commit
`5d74d582610e690954dac328f7fabbd092f57b78`.
The local formal study ran from clean source commit
`6a1ce9a00fb7e1f1e168ec5a2a7491e333fbd522`.
GitHub Actions independently regenerated the full study from source commit
`096b30fbb721a142dd81faa2c7359fbc3b2a4bf8`.
The publication commit is `c77b9acf94603bab045a96c0299ac8853f90a27e`.
Later delivery-only commits do not change the simulation implementation or published values.

## Validation evidence

| Check | Result |
| --- | --- |
| Unit, invariant, randomized, and tooling tests | 81 passed; zero failures or skips locally and in the publication run |
| Clean CI installations | Linux and Windows; Python 3.11 and 3.13 |
| Full experiment design | 955 of 955 trials completed across five seeds |
| Independent artifact checker | All 955 result rows and eight complete iteration traces passed |
| Repeated local full-run command | 955 verified cache hits; zero cache repairs |
| Cross-environment comparison | All 955 trials matched across 80 numeric columns; maximum absolute difference 0.0 |
| Input and actual-arrival hashes | Identical between local and CI runs |
| Core implementation fingerprint | Identical between local and CI runs |
| Statistical artifacts | 3,993 aggregate rows and 2,802 paired comparison rows |
| Bootstrap procedure | 2,000 seed-level resamples; 95% percentile intervals |
| Static figures | Twelve generated SVG figures committed; PNG generation also supported |
| Report generation | Complete tables; no unresolved placeholders or missing-condition values |
| Artifact publication marker | File hashes and source/tooling/configuration fingerprints verified |
| Git workflow | Direct commits to main; no feature branches or pull requests created |

The numeric comparison joined trials on experiment, condition, policy, and seed. It excluded
execution metadata such as commit ID, UTC timestamp, Python version, and host platform.
The result is recorded in [CROSS_ENVIRONMENT_CHECK.json](CROSS_ENVIRONMENT_CHECK.json).
Exact agreement in this comparison is not a promise of bitwise equality on every future platform.

The formal publication CI run is
[34306820938](https://github.com/danial-maqbool/LLM-Serving-Scheduler-Lab/actions/runs/34306820938).
It passed all four test environments before publishing data and rerunning the test suite.
The final delivery commit is checked again by the same CI workflow. Its publication job verifies
existing output hashes and skips unnecessary regeneration when fingerprints match.

The four temporary setup branches were removed only after verifying their commits were ancestors
of main. They must not be recreated. No valid history was force-pushed.

## Commands executed

Installation and tests, baseline smoke, the 12-trial suite, CSV generation, the complete 955-trial
suite, cache reuse, statistical aggregation, all twelve plots, report generation, artifact checks,
and local documentation-link checks were exercised. An additional CSV timing-grid check matched an
affine interpolation oracle. The chat environment used installed dependency packages; clean online
installation was independently exercised by GitHub Actions on both operating systems.

Use [the reproduction guide](REPRODUCIBILITY.md) for commands and file formats.
The Markdown checker validates local files and anchors. It does not claim to test external services.
Known credential patterns were scanned in the tracked code; no credentials were found.
No private documents, model credentials, or user message content were added to this repository.

## Interpretation boundary

All reported latencies and throughput values are **SIMULATED**. No GPU inference measurements
were performed. The chat runtime had no usable CUDA device. The trace adapter's supplied timing
grid is synthetic. See [the hardware calibration plan](CALIBRATION.md).

Finite cohorts and five seeds limit inference. Timing-model uncertainty is not included in bootstrap
intervals. The report distinguishes request-average TPOT from individual token stalls, throughput
with drain from arrival-window throughput, and waiting-time proxies from infinite-stream starvation.
There is no universal best scheduler or chunk size established by this lab.

Read [REPORT.md](../REPORT.md) for the numerical trade-offs, and
[the methodology](METHODOLOGY.md) for exact accounting and model assumptions.
