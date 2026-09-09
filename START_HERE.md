# Start here

This is an executable simulation lab, not an application. Work directly on the existing `main` branch.
Do not create another repository. Do not force-push history or recreate the removed setup branches.

Read [README.md](README.md), [methodology](docs/METHODOLOGY.md), [experiment design](docs/EXPERIMENT_PLAN.md),
and [the audit](docs/AUDIT.md). The original starter is preserved in commit
`5d74d582610e690954dac328f7fabbd092f57b78`.

Start with `python -m pytest -q`. Run the small suite before the full study. Install the analysis extra
for plotting and YAML. Use a clean committed source tree for formal results. Dirty-source runs remain
useful for debugging but must not be presented as release measurements.

Each tested implementation phase is committed and pushed to `main`. The repository's CI checks
Linux and Windows. All numerical timing results must stay labeled **SIMULATED** unless a separate
hardware measurement procedure has actually run. No GPU is required for the lab.
