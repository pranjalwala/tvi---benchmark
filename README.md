# tvi-benchmark

**Hardware-agnostic TVI / Dot-Gain benchmark for halftoning research.**

Implements the physical print-and-scan evaluation layer from the
*Halftoning Benchmark Specification* (Kaur, Das, Wala, Pal, 2025).

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10.

## Quick demo (no printer needed)

```bash
python scripts/generate_targets.py --demo --output_dir demo_data/
python scripts/validate_scan.py demo_data/tone_ramp_K.tif --demo
python scripts/batch_measure_tvi.py --demo --results_dir results/demo/ --fit_yn
```

## With real scans

See **[examples/README.md](examples/README.md)** for:
- Required directory structure
- Scanner settings
- Naming conventions
- Full command reference

## Tests

```bash
pytest tvi/tests/ -v      # 94 tests
```

## Package structure

```
tvi/
  io/                  — TIFF loading and metadata extraction
  calibration_targets/ — Target TIFF generation (tone ramps, step wedge)
  preprocessing/       — Geometric alignment, patch extraction
  calibration/         — Scanner linearisation, printer transfer function
  measurement/         — Murray-Davies, TVI curves, Yule-Nielsen fitting
  aggregation/         — Statistics, CI, CSV export (benchmark Table 3)
  simulation/          — Dot-gain simulator from measured TVI curves
  visualization/       — Matplotlib plots

scripts/
  generate_targets.py      — Generate calibration target TIFFs
  validate_scan.py         — Validate a real scan before measurement
  measure_tvi_from_scan.py — Single-scan TVI pipeline
  batch_measure_tvi.py     — Batch processing across methods/channels
  run_benchmark.py         — Full benchmark entry point
```

## Implemented metrics (Phase 1)

- [x] TVI / Dot Gain (Murray-Davies + Yule-Nielsen)
- [x] Printer transfer function
- [x] Dot-gain simulator
- [x] Confidence intervals (Student's t, 3 sheets × 3 scans)
- [ ] Channel misregistration (Phase 2)
- [ ] Signal-to-banding ratio (Phase 2)
- [ ] Trapping coefficient (Phase 2)
- [ ] Overprint ΔE (Phase 2)
- [ ] Dot loss (Phase 2)

## License

MIT
