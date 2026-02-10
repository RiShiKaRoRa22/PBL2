"""
Configuration file for the Transport Analytics ML/DL System
"""
from pathlib import Path

# =====================================================
# PROJECT PATHS
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# DATA FILES
# =====================================================
TRAIN_FILE = DATA_DIR / "train_engineered.csv"
TEST_FILE = DATA_DIR / "test_engineered.csv"

# =====================================================
# MODEL PARAMETERS
# =====================================================
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2

# =====================================================
# DEEP LEARNING PARAMETERS
# =====================================================
DL_CONFIG = {
    "embedding_dim": 32,
    "hidden_dims": [256, 128, 64],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 100,
    "early_stopping_patience": 15,
    "mc_dropout_samples": 50
}

# =====================================================
# TARGET VARIABLES
# =====================================================
TARGETS = {
    "regression": ["passenger_demand", "load_factor"],
    "classification": ["utilization_encoded"]
}

# =====================================================
# FEATURE GROUPS
# =====================================================
CATEGORICAL_FEATURES = ["route_id", "utilization_encoded"]
NUMERIC_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_peak",
    "speed", "SRI", "congestion_level", "capacity",
    "demand_capacity_ratio", "speed_congestion_ratio",
    "utilization_score", "route_avg_speed", "route_avg_demand",
    "route_avg_congestion"
]

# =====================================================
# EVALUATION METRICS
# =====================================================
REGRESSION_METRICS = ["rmse", "mae", "r2"]
CLASSIFICATION_METRICS = ["accuracy", "f1_weighted", "f1_macro"]
