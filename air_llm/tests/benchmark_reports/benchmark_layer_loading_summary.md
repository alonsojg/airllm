# AtmoLLM Layer Loading Benchmark Summary

## Scope

This benchmark compares baseline AtmoLLM single-layer loading (`max_layers_in_memory=1`) against multi-layer loading (`2`, `4`, `8`) using:

- Model: `garage-bAInd/Platypus2-7B`
- Compression: `4bit`
- Device: `cuda:0`
- Generation workload: `32` generated tokens
- Warmup runs: `1`
- Measured runs per setting: `2`

The underlying detailed outputs are in:

- `benchmark_layer_loading_report.json`
- `benchmark_layer_loading_settings.csv`
- `benchmark_layer_loading_runs.csv`

## Headline Result

Generation throughput improves consistently as more layers are kept resident in memory, but the gains are **sublinear** and show **diminishing returns** as the number of loaded layers increases.

This dataset does **not** support a cubic or square-root scaling claim. A more accurate description is:

> Increasing `max_layers_in_memory` improves generation throughput, but the gain flattens as layer residency grows.

## Results Table

| Requested layers | Effective layers | Mean elapsed time (s) | Mean tokens/s | Speedup vs 1-layer |
|---|---:|---:|---:|---:|
| 1 | 1 | 99.69 | 0.3210 | 1.00x |
| 2 | 2 | 74.04 | 0.4322 | 1.35x |
| 4 | 4 | 58.18 | 0.5500 | 1.71x |
| 8 | 8 | 51.01 | 0.6273 | 1.95x |

## Incremental Gain Per Step

- `1 -> 2` layers: `0.3210 -> 0.4322 tok/s`, about **+34.6%**
- `1 -> 4` layers: `0.3210 -> 0.5500 tok/s`, about **+71.3%**
- `1 -> 8` layers: `0.3210 -> 0.6273 tok/s`, about **+95.4%**

This pattern is the clearest signal in the benchmark: performance improves with more resident layers, but the curve is still clearly sublinear rather than linear or cubic.

## Memory Observations

Peak VRAM allocation increased as more layers stayed resident:

- `1` layer: about `0.84 GB`
- `2` layers: about `0.84 GB`
- `4` layers: about `1.59 GB`
- `8` layers: about `3.09 GB`

So the benchmark shows a direct tradeoff:

- More resident layers reduce repeated load overhead and improve generation speed.
- More resident layers increase memory pressure.

## Safety Behavior

The runtime safety guard was active, but it did not need to clamp any of the tested settings:

- Effective layers matched requested layers for all tested values.
- No init clamp warnings were emitted.
- No runtime clamp warnings were emitted.
- Detected safe limit was `16` to `17` layers in this environment.

That means `1`, `2`, `4`, and `8` were all comfortably below the 80% memory safety ceiling for this machine.

## Conclusion

For this setup, multi-layer residency clearly outperforms original single-layer loading.

The strongest concise conclusion is:

> Moving from single-layer loading to 8-layer residency improved generation throughput by about **1.95x**, with diminishing incremental gains at higher residency levels.

A defensible interpretation for a report is:

> AtmoLLM benefits materially from loading multiple layers at once, but the scaling is sublinear rather than proportional to the number of resident layers.

## Suggested Report Language

AtmoLLM was benchmarked on `garage-bAInd/Platypus2-7B` with 4-bit compression on a CUDA GPU while varying `max_layers_in_memory` across `1`, `2`, `4`, and `8`. Relative to baseline single-layer loading, throughput improved monotonically from `0.321 tok/s` to `0.627 tok/s`, yielding an overall `1.95x` speedup at `8` layers. The speedup curve was sublinear, with diminishing returns at higher residency levels, indicating that keeping more layers resident reduces load overhead but does not scale proportionally with the number of retained layers. Peak VRAM usage rose from roughly `0.84 GB` at `1` layer to `3.09 GB` at `8` layers, showing the expected performance-memory tradeoff.
