# Project specification

Build a small research-engineering artifact, not a web app or a paper submission.
Study how monolithic prefill, continuous batching, and chunked prefill interact under explicit
resource limits and alternative analytical timing models.

Required deliverables are implemented in source modules rather than notebook-only code:

| Area | Contract |
| --- | --- |
| Simulation | Validated lifecycle, hard limits, deterministic execution, complete request accounting |
| Policies | Serial FCFS, monolithic continuous batching, chunked prefill; explicit priority variants |
| Timing | Analytical, piecewise, and trace interpolation with declared provenance |
| Workloads | Poisson, burst, closed-loop, length regimes, adversarial ordering, CSV replay |
| Metrics | Request and token-gap latency, throughput, finite-window backlog, variation, fairness proxies |
| Experiments | Fixed design, matched seeds, verified caching, restartable trials, provenance |
| Analysis | Seed-level intervals, paired comparisons, static figures, measured/simulated distinction |
| Delivery | Tests, CI, report, executable commands, readable documentation, main-only Git history |

The supplied starter remains in Git history. [AUDIT.md](AUDIT.md) explains corrections.
[METHODOLOGY.md](METHODOLOGY.md) defines the final semantics.
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) specifies the experiment matrix.

No external LLM API, dashboard, cloud deployment, or fabricated hardware result is permitted.
No coefficient may depend on the scheduler's identity. Correctness tests must not enforce a preferred
performance ranking. Cost-model limitations remain part of the result, not defects to conceal.
