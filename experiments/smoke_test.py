import imageio.v3 as iio
import numpy as np
from pathlib import Path

from tvi.simulation import (
    SimulatorParams,
    simulate_dot_gain,
)

img = iio.imread(
    "dataset/Copy of Copy of kodim17.png"
)

if img.ndim == 3:
    img = img.mean(axis=2)

img = img.astype(np.float32)
img /= img.max()

# crude thresholding
img2 = img.copy()

h, w = img2.shape
halftone = np.zeros_like(img2, dtype=np.uint8)

for y in range(h):
    for x in range(w):

        old = img2[y, x]
        new = 1.0 if old > 0.5 else 0.0

        halftone[y, x] = new
        err = old - new

        if x + 1 < w:
            img2[y, x + 1] += err * 7.0 / 16.0

        if y + 1 < h:
            if x > 0:
                img2[y + 1, x - 1] += err * 3.0 / 16.0

            img2[y + 1, x] += err * 5.0 / 16.0

            if x + 1 < w:
                img2[y + 1, x + 1] += err * 1.0 / 16.0

img2 = np.clip(img2, 0.0, 1.0)

params = SimulatorParams(
    r_mech=0.8,
    n_yn=1.6,
    r_paper=1.0,
    r_ink=0.0,
    channel="K",
    method="SmokeTest",
    printer_label="VirtualPrinter",
    rmse=0.0,
)
simulated = simulate_dot_gain(
    halftone,
    params,
)

Path("results/simulated").mkdir(
    parents=True,
    exist_ok=True,
)

iio.imwrite(
    "results/simulated/kodim17_halftone.png",
    (halftone * 255).astype(np.uint8),
)

iio.imwrite(
    "results/simulated/kodim17_simulated.png",
    (simulated * 255).astype(np.uint8),
)

print("Done.")