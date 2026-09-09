# Reproduce the study

## Install and test

Use Python 3.11 or newer. The core simulator uses only the standard library.
The `dev` extra installs pytest. The `analysis` extra installs NumPy, pandas, Matplotlib, and PyYAML.
The CI matrix tests Python 3.11 and 3.13 on Linux and Windows.

```bash
python -m pip install -e ".[dev,analysis]"
python -m pytest -q
python scripts/check_links.py
```

The committed dependency bounds permit clean installations on these Python versions. Each run records
Python/platform metadata. Dependency versions used for artifact publication are saved in
`results/release/environment.json`. No claim of bitwise identity across all library versions is made.

## Small run

```bash
python -m llm_scheduler_lab run --config configs/smoke.json --output results/smoke
python -m llm_scheduler_lab summarize --results results/smoke --output results/smoke/statistics
python -m llm_scheduler_lab plot --results results/smoke --output results/smoke/figures
python scripts/check_artifacts.py results/smoke
```

This runs twelve small trials. The smoke plot is not a replacement for the full study.
Use a clean committed checkout for formal runs. With an unavailable Git checkout, use `--allow-dirty`
only for exploratory work; the result records unavailable/dirty provenance and fails release checks.

## Full run and resume

```bash
python -m llm_scheduler_lab run --config configs/study.json --output results/reproduction
python -m llm_scheduler_lab summarize --results results/reproduction --output results/reproduction/statistics
python -m llm_scheduler_lab plot --results results/reproduction --output results/reproduction/figures
python scripts/check_artifacts.py results/reproduction
```

Repeat the `run` command unchanged to verify all cache entries and reuse completed trials. A killed
run retains atomic per-trial caches. The runner checks cache payload hashes, implementation fingerprint,
schema, workload, seed, policy, timing parameters, and external file content. Corrupt entries are
recomputed. A dirty-source cache cannot certify a clean-source run. In-memory state is never shared
between policies.

Use `--experiment E5` to run one experiment. The complete plot/report templates expect the complete
study; single-experiment runs still produce machine-readable metrics and statistical summaries.
The full report builder intentionally refuses incomplete study inputs.

## CSV and YAML

Generate a synthetic CSV:

```bash
python -m llm_scheduler_lab trace --output results/workload.csv --seed 101 --requests 128 --mode poisson --regime mixed
```

The CSV header is exactly `request_id,arrival_ms,prompt_tokens,output_tokens`. IDs must be unique;
arrival times must be finite and nonnegative; lengths must be positive integers. Order is normalized
by arrival and ID. Set `input_trace` in an experiment to a path relative to its config directory.
JSON and YAML use the same suite schema. YAML requires the analysis extra. Fixed-concurrency CSVs
contain zero-time templates and require `arrival_mode: fixed_concurrency` in the workload config.

## Artifacts and evidence

Each trial records its complete configuration, input and actual-arrival hashes, seed, source commit,
core-source hash, timestamp, environment, cost-model provenance, and metrics. Every simulation output
is labeled `SIMULATED`, including predictions from a measured timing grid.

`summary.csv` contains per-run metrics. `runs.json` contains trial parameters, cost models, and
iteration histograms. `manifest.json` hashes both. `statistics/` contains aggregates, paired changes,
and resampling rules. Selected E5 traces retain exact per-request allocations and timestamps.
Large `.runs` caches remain local; the release retains summary evidence and selected traces.

Strictly compare metric columns or canonical run configuration, not UTC timestamps or host platform.
The same source and workload are exactly replayable in the same Python environment. Floating-point
and dependency differences across environments should be checked with numerical tolerance.

## Publishing and Git safety

The push-triggered CI first runs all four test environments. Only then can its publication job run.
That job uses repository-scoped `GITHUB_TOKEN` permissions and commits only the generated release
artifacts, figures, README, and report. It never accesses an external model API or personal data.

The publisher skips regeneration when simulation/tooling/config fingerprints and artifact hashes
match. It checks that `main` has not advanced before pushing. A conflicting push fails rather than
rewriting history. No feature branches or PRs are created. Generated data commits are followed by
validation on the final repository head during delivery.

The source commit recorded by a simulation precedes its data commit. That is intentional: a commit
cannot contain files whose metadata depends on that same commit's hash. Later documentation-only
commits may retain an earlier simulation source commit when its implementation hash is unchanged.
