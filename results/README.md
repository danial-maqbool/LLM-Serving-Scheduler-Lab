# Result artifacts

`release/` holds the published simulation study: per-run CSV, complete run/configuration metadata,
checksums, seed-level statistics, selected head-of-line traces, and hardware-capability metadata.
These are synthetic experiments, not GPU measurements. The repository CI regenerates this directory
from a clean source commit before publishing it.

Other directories in `results/` are ignored scratch output. Per-trial `.runs/` caches are always ignored.
Use a new output directory for a new experiment. Do not edit a result table manually.
