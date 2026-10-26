"""Train the ANN rainfall-estimation model and save the model + scaler.

Run from the project folder:   python model/train_model.py
"""
import os

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "rainfall_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# The feature ORDER must be identical in app.py
FEATURES = ["temperature", "humidity", "pressure", "wind_speed", "cloud_cover"]
TARGET = "rainfall_mm"

# 1. Load data
df = pd.read_csv(DATA_PATH)
X = df[FEATURES].values
y = df[TARGET].values

# 2. Split and scale (fit the scaler on training data only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 3. Build the ANN (regression: output = rainfall in mm)
model = keras.Sequential([
    layers.Input(shape=(len(FEATURES),)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 4. Train
model.fit(
    X_train_s, y_train,
    validation_split=0.2,
    epochs=60,
    batch_size=32,
    verbose=1,
)

# 5. Evaluate
pred = model.predict(X_test_s, verbose=0).flatten().clip(min=0)
print(f"Test MAE : {mean_absolute_error(y_test, pred):.2f} mm")
print(f"Test R2  : {r2_score(y_test, pred):.3f}")

# 6. Save
model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print("Saved:", MODEL_PATH)
print("Saved:", SCALER_PATH)