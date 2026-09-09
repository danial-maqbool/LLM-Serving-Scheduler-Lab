# Methodology

## Scope

The first version is a **discrete-iteration analytical simulator**, not a GPU kernel simulator. It studies scheduling effects under a transparent cost model. The timing coefficients are synthetic unless explicitly calibrated to measured hardware traces.

## Request lifecycle

Each request has:

1. arrival;
2. prefill waiting;
3. prefill service;
4. decode service, one generated token per selected decode iteration;
5. completion.

A request cannot decode before prefill finishes.

## Timing model

For an iteration:

```text
T_iter = T_fixed
       + N_prefill_tokens * C_prefill
       + N_decode_sequences * C_decode
       + N_decode_sequences^2 * C_batch_pressure
```

This is intentionally simple. The next implementation phase should add alternative timing models and calibration hooks rather than pretending the equation is universally accurate.

## Policy semantics

### FCFS

Oldest request is served alone. Its prefill is monolithic, then its output is decoded to completion.

### Continuous batching

Decode-ready requests share an iteration. One waiting prompt can be prefetched monolithically. This creates long iterations when prompts are large.

### Chunked prefill

Decode-ready requests are scheduled first. Remaining iteration capacity is used for bounded prefill chunks. Chunk size is an experimental variable.

## Metrics

### TTFT

```text
TTFT = timestamp(first output token) - arrival timestamp
```

### TPOT

For output length > 1:

```text
TPOT = (completion - first_token) / (output_tokens - 1)
```

### End-to-end latency

```text
E2E = completion - arrival
```

### Iteration variation

Report mean, standard deviation, and coefficient of variation. High iteration-time variance is a useful proxy for scheduler imbalance, but must not be mislabeled as measured pipeline bubbles unless an explicit pipeline model is implemented.

## Reporting discipline

Every figure/table must state:

- simulator version/commit;
- workload seed;
- policy parameters;
- timing model and coefficients;
- request count;
- whether results are simulated or measured.
