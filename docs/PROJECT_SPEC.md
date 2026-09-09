# Project Specification

## Goal

Build a technically defensible mini research-engineering repository that explains and demonstrates the scheduling trade-offs between monolithic prefill, continuous batching, and chunked prefill in LLM serving.

This is **portfolio technical work**, not a research-paper submission and not a UI/product project.

## Definition of done

The project is complete when it has:

- a correct, deterministic simulator with explicit assumptions;
- at least three scheduler families;
- workload generation plus trace import support;
- reproducible experiment configs;
- unit, invariant, regression, and adversarial tests;
- multi-seed sweeps;
- CSV/JSON result artifacts;
- publication-quality plots;
- statistical summaries with uncertainty where appropriate;
- an ablation/sensitivity study;
- a concise technical report explaining results and limitations;
- CI passing from a clean checkout;
- no unsupported claims about real GPU performance.

## Priority implementation work

1. Refactor timing into pluggable cost models.
2. Add per-iteration token-budget and KV/batch-capacity invariants.
3. Add trace import/export.
4. Add experiment sweep runner with deterministic seeds.
5. Add pandas/matplotlib analysis scripts.
6. Add bootstrap confidence intervals for aggregate comparisons.
7. Add timeline/Gantt-style scheduler visualization (static plot only, no frontend).
8. Add saturation and adversarial workloads.
9. Add scheduler fairness metrics.
10. Write `REPORT.md` with methodology, results, caveats, and reproducibility instructions.

## Non-goals

- No web application.
- No dashboard.
- No LLM API integration.
- No pretending analytical timings are hardware measurements.
- No copying vLLM/Sarathi implementations wholesale.
- No weakening tests to force expected narratives.

## Quality bar

The repository should look like a small systems lab: clear hypotheses, explicit models, controlled experiments, reproducibility, plots, and honest limitations.
