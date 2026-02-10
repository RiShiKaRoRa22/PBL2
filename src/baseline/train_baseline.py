import pandas as pd
import numpy as np


from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix
)

# =========================
# LOAD DATA
# =========================
TRAIN_PATH = "data/final/train_engineered.csv"
TEST_PATH  = "data/final/test_engineered.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# =========================
# FEATURE SET (BASELINE)
# =========================
FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_peak",
    "speed",
    "congestion_level",
    "capacity",
    "route_avg_speed",
    "route_avg_demand",
    "route_avg_congestion"
]

X_train = train_df[FEATURES]
X_test  = test_df[FEATURES]

# =========================
# BASELINE 1: PASSENGER DEMAND
# =========================
y_train = train_df["passenger_demand"]
y_test  = test_df["passenger_demand"]

demand_model = LGBMRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

demand_model.fit(X_train, y_train)

train_pred = demand_model.predict(X_train)
test_pred  = demand_model.predict(X_test)

print("\n=== Passenger Demand (LightGBM Baseline) ===")
print(f"Train MAE: {mean_absolute_error(y_train, train_pred):.3f}")
print(f"Test  MAE: {mean_absolute_error(y_test, test_pred):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.3f}")
print(f"Test R2  : {r2_score(y_test, test_pred):.3f}")

# =========================
# BASELINE 2: LOAD FACTOR
# =========================
y_train = train_df["load_factor"]
y_test  = test_df["load_factor"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

load_model = Ridge(alpha=1.0)
load_model.fit(X_train_scaled, y_train)

train_pred = load_model.predict(X_train_scaled)
test_pred  = load_model.predict(X_test_scaled)

print("\n=== Load Factor (Ridge Baseline) ===")
print(f"Train MAE: {mean_absolute_error(y_train, train_pred):.3f}")
print(f"Test  MAE: {mean_absolute_error(y_test, test_pred):.3f}")
print(f"Test R2  : {r2_score(y_test, test_pred):.3f}")

# =========================
# BASELINE 3: UTILIZATION STATUS
# =========================
y_train = train_df["utilization_encoded"]
y_test  = test_df["utilization_encoded"]

util_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

util_model.fit(X_train_scaled, y_train)

train_pred = util_model.predict(X_train_scaled)
test_pred  = util_model.predict(X_test_scaled)

print("\n=== Utilization Status (Logistic Baseline) ===")
print(f"Accuracy : {accuracy_score(y_test, test_pred):.3f}")
print(f"Macro F1 : {f1_score(y_test, test_pred, average='macro'):.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
