# TVI Benchmark — Usage Guide

## Quick start (no printer needed)

```bash
# 1. Install
pip install -e .

# 2. Generate demo calibration targets
python scripts/generate_targets.py --demo --output_dir demo_data/

# 3. Validate the generated TIFF
python scripts/validate_scan.py demo_data/tone_ramp_K.tif --demo

# 4. Run the full batch benchmark with synthetic data
python scripts/batch_measure_tvi.py --demo --results_dir results/demo/ --fit_yn
```

---

## Directory structure

```
tvi-benchmark/
│
├── configs/
│   └── device_template.yaml      ← Copy and edit this for your lab
│
├── generated_targets/            ← Run generate_targets.py to populate
│   ├── tone_ramp_C.tif
│   ├── tone_ramp_M.tif
│   ├── tone_ramp_Y.tif
│   ├── tone_ramp_K.tif
│   ├── step_wedge.tif
│   ├── overprint_patches.tif
│   └── manifest.json
│
├── scans/                        ← YOU supply these (your real scans)
│   ├── wedges/
│   │   └── session1_wedge.tif
│   ├── DBS/
│   │   ├── K/
│   │   │   ├── sheet1_scan1.tif
│   │   │   ├── sheet1_scan2.tif
│   │   │   ├── sheet1_scan3.tif
│   │   │   ├── sheet2_scan1.tif
│   │   │   └── ...
│   │   └── C/
│   │       └── ...
│   ├── ErrorDiffusion/
│   │   └── ...
│   └── OrderedDither/
│       └── ...
│
├── results/
│   ├── csv/
│   │   ├── tvi_metric_table.csv  ← Benchmark Table 3
│   │   ├── tvi_full_curves.csv
│   │   └── yule_nielsen_n.csv
│   ├── plots/
│   │   ├── tvi_curves.png
│   │   ├── tvi_scalar.png
│   │   └── transfer_functions.png
│   └── curves/
│       └── transfer_<method>_<channel>.csv
│
├── demo_data/                    ← Synthetic demo TIFFs for smoke testing
│   ├── tone_ramp_K.tif
│   └── ...
│
├── tvi/                          ← Python package
├── scripts/                      ← Entry-point scripts
└── tests/                        ← pytest suite
```

---

## What images to provide

### Step 1 — Print the calibration targets

Generate the targets for your specific printer:

```bash
# Edit configs/device_template.yaml for your printer/scanner first
python scripts/generate_targets.py --config configs/my_lab.yaml --output_dir generated_targets/
```

Send these files to your printer:
- `generated_targets/tone_ramp_C.tif`
- `generated_targets/tone_ramp_M.tif`
- `generated_targets/tone_ramp_Y.tif`
- `generated_targets/tone_ramp_K.tif`
- `generated_targets/step_wedge.tif`

Print each tone ramp **three times** (three separate sheets) per halftoning method.
Print the step wedge **on every sheet** alongside the tone ramp (co-printed reference).

### Step 2 — Scan the printed sheets

Scan each printed sheet **three times** (three separate scan passes, fixed placement).

---

## Naming convention for scan files

```
scans/<method>/<channel>/sheet<N>_scan<M>.tif
```

- `<method>` must match one of the halftoning method names you care about  
  (e.g. `DBS`, `ErrorDiffusion`, `OrderedDither`, `DeepLearning`)
- `<channel>` is the ink channel letter: `C`, `M`, `Y`, or `K`
- `N` = sheet replicate number (1, 2, 3)
- `M` = scan replicate number (1, 2, 3)

Examples:
```
scans/DBS/K/sheet1_scan1.tif
scans/DBS/K/sheet1_scan2.tif
scans/DBS/K/sheet1_scan3.tif
scans/DBS/K/sheet2_scan1.tif
...
scans/DBS/K/sheet3_scan3.tif     ← total 9 TIFFs per (method, channel)

scans/wedges/session1_wedge.tif  ← one step-wedge scan per session
```

---

## Required TIFF settings (scanner)

| Setting | Requirement |
|---------|-------------|
| Format | TIFF (uncompressed or lossless LZW/Deflate) |
| Bit depth | 8-bit or 16-bit per channel (match `scanner.bit_depth` in config) |
| Colour space | RGB (not grayscale; we extract specific channels per ink) |
| Resolution | Match `scanner.capture_dpi` in your config |
| Automatic corrections | **ALL OFF** — no auto-levels, no colour management, no sharpening |
| Colour profiles | Embedded profiles OK but disable automatic application |
| JPEG compression | Never use JPEG for measurement scans |

---

## Scanner settings checklist

Before scanning:

- [ ] Disable auto-exposure / auto-levels
- [ ] Disable automatic colour correction
- [ ] Disable sharpening / unsharp mask
- [ ] Disable descreen / dust removal filters
- [ ] Set white balance to fixed (not auto)
- [ ] Use the same glass placement for all replicate scans (fix paper position)
- [ ] Warm up the scanner lamp for at least 5 minutes before the session
- [ ] Scan the step wedge in the same session as the tone ramps

---

## Config file

Copy `configs/device_template.yaml` and edit:

```yaml
scanner:
  label: "MyScanner"
  capture_dpi: 1200          # must match your scan resolution
  bit_depth: 16              # 8 or 16

printer:
  label: "MyPrinter"
  print_dpi: 1200
  channels: [C, M, Y, K]
```

---

## Commands

### Generate calibration targets
```bash
python scripts/generate_targets.py --config configs/my_lab.yaml
```

### Validate a scan before measurement
```bash
python scripts/validate_scan.py scans/DBS/K/sheet1_scan1.tif \
    --config configs/my_lab.yaml \
    --save_diagnostic results/diagnostics/check.png
```

### Measure TVI from a single scan
```bash
python scripts/measure_tvi_from_scan.py \
    --scan scans/DBS/K/sheet1_scan1.tif \
    --channel K \
    --method DBS \
    --config configs/my_lab.yaml \
    --wedge scans/wedges/session1_wedge.tif \
    --fit_yn \
    --results_dir results/single/
```

### Full batch benchmark
```bash
python scripts/batch_measure_tvi.py \
    --config configs/my_lab.yaml \
    --scans_dir scans/ \
    --results_dir results/ \
    --fit_yn \
    --simulate
```

### Run all tests
```bash
pytest tvi/tests/ -v
```

### Demo (no printer or scanner needed)
```bash
python scripts/generate_targets.py --demo --output_dir demo_data/
python scripts/validate_scan.py demo_data/tone_ramp_K.tif --demo
python scripts/batch_measure_tvi.py --demo --results_dir results/demo/ --fit_yn
```

---

## Output files explained

| File | Content |
|------|---------|
| `results/csv/tvi_metric_table.csv` | Primary scalar: TVI at 50% ± 95% CI per method/channel |
| `results/csv/tvi_full_curves.csv` | Full TVI(a_nom) curve, long format |
| `results/csv/yule_nielsen_n.csv` | Fitted Yule-Nielsen n factor |
| `results/curves/transfer_*.csv` | Printer transfer function a_nom→a_print |
| `results/plots/tvi_curves.png` | TVI(a_nom) curves with CI shading |
| `results/plots/tvi_scalar.png` | Bar chart of TVI at 50% |
| `results/plots/transfer_functions.png` | Transfer function overlay |

---

## Troubleshooting

**"No manifest found in TIFF"**  
Your scan does not contain the embedded patch manifest.  This is expected for  
raw scanner output — you must supply a manifest JSON file that describes where  
each patch is in the scan.  Use `validate_scan.py` to visualise patch positions  
and `measure_tvi_from_scan.py --no_wedge` with a hand-edited manifest.

**"Fiducial detection failed"**  
The four corner marks were not found.  Check that:  
- The scan includes the full printed area including margins  
- Auto-levels was not applied (which clips the dark marks)  
- The scanner DPI matches `scanner.capture_dpi` in the config

**"DPI mismatch warning"**  
Update `scanner.capture_dpi` in your config YAML to match the actual scan DPI.

**"TVI is negative"**  
This indicates either (a) dot *shrinkage* rather than gain (rare but possible  
on electrophotographic systems), or (b) a scanner calibration error.  Run  
`validate_scan.py` and inspect the histogram — if the image is very bright  
or very dark, the calibration needs a real step wedge.

---

## Citation

If you use this benchmark, please cite the benchmark specification:

> Kaur, Das, Wala, Pal. *Halftoning Benchmark Specification*. 2025.
