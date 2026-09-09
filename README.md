# LLM Serving Scheduler Lab

A compact research-engineering lab for studying how LLM serving schedulers trade off **time-to-first-token (TTFT)**, **time-per-output-token (TPOT)**, end-to-end latency, throughput, and iteration-time variance under mixed prefill/decode workloads.

This repository is intentionally **not an application**. It is an experiment-first systems project: simulator, scheduling policies, workloads, metrics, ablations, plots, tests, and a technical report.

## Core question

How do scheduling choices change latency and throughput when long compute-heavy prefills compete with memory-sensitive decode steps?

The first study compares:

1. **FCFS / request-at-a-time** baseline
2. **Continuous batching** with monolithic prefills
3. **Chunked prefill** with configurable chunk sizes

The simulator is deliberately explicit about its assumptions. It is not presented as a cycle-accurate GPU simulator. The project should eventually include sensitivity analysis and, where hardware is available, calibration against measured inference traces.

## Primary metrics

- TTFT: arrival to first generated token
- TPOT: average time between generated output tokens after the first token
- End-to-end latency
- Request throughput
- Token throughput
- P50 / P95 / P99 latency
- Iteration-time mean, variance, and coefficient of variation
- Decode stall time
- Prefill waiting time
- Scheduler fairness / starvation indicators

## Repository layout

```text
LLM-Serving-Scheduler-Lab/
├── src/llm_scheduler_lab/
│   ├── models.py
│   ├── workload.py
│   ├── simulator.py
│   ├── metrics.py
│   └── policies/
│       ├── base.py
│       ├── fcfs.py
│       ├── continuous_batching.py
│       └── chunked_prefill.py
├── scripts/
│   └── run_experiment.py
├── configs/
│   └── baseline.json
├── tests/
├── docs/
│   ├── METHODOLOGY.md
│   ├── EXPERIMENT_PLAN.md
│   └── PROJECT_SPEC.md
├── results/
├── figures/
└── .github/workflows/ci.yml
```

## Quick start

```bash
python -m pip install -e .[dev]
pytest -q
python scripts/run_experiment.py --config configs/baseline.json --output results/baseline.csv
```

## Current foundation

The starter implementation already provides:

- deterministic request/state models;
- a synthetic workload generator;
- an event/iteration-based simulator;
- FCFS, continuous-batching, and chunked-prefill policies;
- TTFT/TPOT/E2E/throughput metrics;
- a CLI experiment runner;
- smoke/regression tests;
- CI;
- a detailed experiment and validation plan.

The next phase should deepen the timing model, add experiment sweeps and plots, validate scheduler invariants, add adversarial workloads, and produce a technically defensible report.

## Research integrity

Results must distinguish between:

- **simulated conclusions** under the stated model;
- **measured hardware results**, if added later;
- **claims from prior work**.

Do not describe simulated timings as GPU measurements.

## License

MIT.
