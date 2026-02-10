"""
Create sample data for testing the ML pipeline
This generates synthetic data matching the expected format
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Create directories
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_train = 10000
n_test = 2500

def generate_dataset(n_samples):
    """Generate synthetic transport data"""
    
    data = {
        # Time features
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'is_peak': np.random.randint(0, 2, n_samples),
        
        # Traffic features (normalized)
        'speed': np.random.uniform(-1, 1, n_samples),
        'SRI': np.random.uniform(-1, 1, n_samples),
        'congestion_level': np.random.uniform(-1, 1, n_samples),
        
        # Fleet features (normalized)
        'capacity': np.random.uniform(-1, 1, n_samples),
        
        # Derived features (normalized)
        'demand_capacity_ratio': np.random.uniform(-1, 1, n_samples),
        'speed_congestion_ratio': np.random.uniform(-1, 1, n_samples),
        'utilization_score': np.random.uniform(-1, 1, n_samples),
        
        # Route aggregates (normalized)
        'route_avg_speed': np.random.uniform(-1, 1, n_samples),
        'route_avg_demand': np.random.uniform(-1, 1, n_samples),
        'route_avg_congestion': np.random.uniform(-1, 1, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate targets based on features (with some correlation)
    # Passenger demand (normalized, 0-1 range after denormalization)
    df['passenger_demand'] = (
        0.3 * df['is_peak'] + 
        0.2 * (1 - df['congestion_level']) + 
        0.3 * df['route_avg_demand'] +
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Load factor (normalized, 0-1 range)
    df['load_factor'] = (
        0.4 * df['passenger_demand'] + 
        0.3 * df['demand_capacity_ratio'] +
        0.2 * df['is_peak'] +
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Utilization status (0: Underutilized, 1: Optimal, 2: Overloaded)
    load_factor_raw = df['load_factor'].values
    utilization = np.zeros(n_samples, dtype=int)
    utilization[load_factor_raw < -0.3] = 0  # Underutilized
    utilization[(load_factor_raw >= -0.3) & (load_factor_raw < 0.5)] = 1  # Optimal
    utilization[load_factor_raw >= 0.5] = 2  # Overloaded
    df['utilization_encoded'] = utilization
    
    # Add utilization_status text (for reference)
    status_map = {0: 'Underutilized', 1: 'Optimal', 2: 'Overloaded'}
    df['utilization_status'] = df['utilization_encoded'].map(status_map)
    
    return df

# Generate train and test datasets
print("Generating sample training data...")
train_df = generate_dataset(n_train)
train_df.to_csv("data/processed/train_engineered.csv", index=False)
print(f"✓ Created: data/processed/train_engineered.csv ({n_train} samples)")

print("\nGenerating sample test data...")
test_df = generate_dataset(n_test)
test_df.to_csv("data/processed/test_engineered.csv", index=False)
print(f"✓ Created: data/processed/test_engineered.csv ({n_test} samples)")

print("\n" + "="*60)
print("SAMPLE DATA CREATED SUCCESSFULLY")
print("="*60)
print("\nDataset info:")
print(f"  Train samples: {n_train}")
print(f"  Test samples: {n_test}")
print(f"  Features: {len(train_df.columns) - 3}")  # Exclude targets
print(f"  Targets: passenger_demand, load_factor, utilization_encoded")

print("\nSample data preview:")
print(train_df.head())

print("\nTarget distribution:")
print(train_df['utilization_encoded'].value_counts().sort_index())

print("\n" + "="*60)
print("You can now run: python main_pipeline.py")
print("="*60)
