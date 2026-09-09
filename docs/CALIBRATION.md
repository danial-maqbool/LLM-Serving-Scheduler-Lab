# Hardware calibration interface

Hardware calibration was not executed in the chat environment. No usable CUDA device was detected.
The repository runs without PyTorch, CUDA, an LLM API, or model weights.

Run the capability probe separately:

```bash
python scripts/check_hardware.py --output results/hardware.json
```

This command detects capability. It does not benchmark a model. A CPU-only PyTorch installation
must not be reported as a GPU measurement. The study uses the trace adapter's synthetic demonstration.

## JSON timing grid

A minimal schema example is:

```json
{
  "source_kind": "synthetic",
  "description": "Schema example only, not hardware measurements",
  "rows": [
    {"prefill_tokens": 0, "decode_sequences": 0, "duration_ms": 0.1},
    {"prefill_tokens": 0, "decode_sequences": 2, "duration_ms": 0.6},
    {"prefill_tokens": 128, "decode_sequences": 0, "duration_ms": 2.4},
    {"prefill_tokens": 128, "decode_sequences": 2, "duration_ms": 2.9}
  ]
}
```

Use a cost-model configuration with `name: trace` and `path: relative/path/to/grid.json`.
Paths resolve relative to the experiment configuration. A CSV requires columns `prefill_tokens`,
`decode_sequences`, `duration_ms` and a sibling `grid.metadata.json` file containing provenance.
Do not use the tiny schema example with out-of-range batches. Such batches intentionally fail.

## Future measurement procedure

Select a specific model revision, precision, device, serving engine revision, and attention backend.
Record context length, sequence count, chunk size, warm-up count, synchronization method, clocks,
and sampling count. Measure complete iteration latency after warm-up. Repeat every timing-grid point.
Keep raw samples and uncertainty, not only a fitted curve. Hold out grid points for interpolation
validation. Compare complete request traces against predicted latencies on held-out workloads.

The current two-axis adapter cannot represent context length, KV layout, or prompt count independently.
Add those axes or restrict the experiment domain before claiming calibrated predictions. Record
model/driver/library versions and GPU memory use. Results from a measured grid remain **simulated**
scheduler results; only the timing samples themselves are **measured**.
