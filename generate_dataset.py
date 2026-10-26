
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "dataset.csv")

rng = np.random.default_rng(42)
N = 3000

temperature = rng.uniform(18, 38, N)
humidity = rng.uniform(30, 100, N)
pressure = rng.uniform(990, 1020, N)
wind_speed = rng.uniform(0, 40, N)
cloud_cover = rng.uniform(0, 100, N)

score = (
    0.06 * (humidity - 70)
    + 0.12 * (1008 - pressure)
    + 0.03 * wind_speed
    + 0.025 * (cloud_cover - 50)
    - 0.05 * (temperature - 28)
    + rng.normal(0, 0.6, N)
)
rainfall_mm = np.clip(score, 0, None) ** 2 * 4

df = pd.DataFrame({
    "temperature": temperature.round(1),
    "humidity": humidity.round(1),
    "pressure": pressure.round(1),
    "wind_speed": wind_speed.round(1),
    "cloud_cover": cloud_cover.round(1),
    "rainfall_mm": rainfall_mm.round(2),
})
df.to_csv(OUT_PATH, index=False)

print(f"Saved {len(df)} rows to {OUT_PATH}")
print(f"Rainy rows (> 0.1 mm): {(df['rainfall_mm'] > 0.1).mean() * 100:.1f}%")