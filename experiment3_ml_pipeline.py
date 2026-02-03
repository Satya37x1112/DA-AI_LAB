"""
EXPERIMENT 3: MINI ML PIPELINE (REPRODUCIBLE SCRIPT)
====================================================
This script demonstrates a basic MLOps pipeline with:
- Configuration-driven execution
- Reproducible results
- Model and metrics persistence
- Clear logging at each stage
"""

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EXPERIMENT 3: MINI ML PIPELINE (REPRODUCIBLE SCRIPT)")
print("=" * 80)
print()

# ============================================================
# STAGE 1: LOAD CONFIGURATION
# ============================================================
print("STAGE 1: LOAD CONFIGURATION")
print("-" * 80)

config_file = 'config.json'
print(f"   Loading configuration from: {config_file}")

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
    print(f"   ✓ Configuration loaded successfully")
except FileNotFoundError:
    print(f"   ✗ Error: Configuration file '{config_file}' not found!")
    exit(1)
except json.JSONDecodeError:
    print(f"   ✗ Error: Invalid JSON format in '{config_file}'!")
    exit(1)

# Display loaded configuration
print(f"\n   Configuration Parameters:")
print(f"      Dataset path: {config['dataset_path']}")
print(f"      Target column: {config['target_column']}")
print(f"      Test size: {config['test_size'] * 100}%")
print(f"      Random state: {config['random_state']}")
print(f"      Model type: {config['model_type']}")
print(f"      Output model: {config['output_model_path']}")
print(f"      Output metrics: {config['output_metrics_path']}")
print(f"      Missing value strategy: {config['preprocessing']['missing_value_strategy']}")

# ============================================================
# STAGE 2: LOAD DATA
# ============================================================
print("\n" + "=" * 80)
print("STAGE 2: LOAD DATA")
print("-" * 80)

print(f"\n   Loading dataset from: {config['dataset_path']}")

try:
    df = pd.read_csv(config['dataset_path'])
    print(f"   ✓ Dataset loaded successfully")
except FileNotFoundError:
    print(f"   ✗ Error: Dataset file '{config['dataset_path']}' not found!")
    exit(1)

# Log dataset information
print(f"\n   Dataset Information:")
print(f"      Shape: {df.shape}")
print(f"      Rows: {df.shape[0]}")
print(f"      Columns: {df.shape[1]}")
print(f"      Feature columns: {[col for col in df.columns if col != config['target_column']]}")
print(f"      Target column: {config['target_column']}")

print(f"\n   First few rows:")
print(df.head())

# Check target column exists
if config['target_column'] not in df.columns:
    print(f"   ✗ Error: Target column '{config['target_column']}' not found in dataset!")
    exit(1)

# Display target distribution
print(f"\n   Target column distribution:")
print(df[config['target_column']].value_counts())

# ============================================================
# STAGE 3: PREPROCESSING
# ============================================================
print("\n" + "=" * 80)
print("STAGE 3: PREPROCESSING")
print("-" * 80)

print("\n   Step 3.1: Check for missing values")
missing_values = df.isnull().sum()
print(missing_values)
total_missing = missing_values.sum()

# Handle missing values
if total_missing > 0:
    print(f"\n   Step 3.2: Handle missing values")
    print(f"      Found {total_missing} missing values")
    print(f"      Strategy: {config['preprocessing']['missing_value_strategy']}")
    
    strategy = config['preprocessing']['missing_value_strategy']
    
    for col in df.columns:
        if col != config['target_column'] and df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            else:
                fill_value = df[col].mean()  # Default to mean
            
            df[col].fillna(fill_value, inplace=True)
            print(f"         - Filled '{col}' missing values with {strategy}: {fill_value:.2f}")
    
    print(f"\n   Step 3.3: Verify missing values handled")
    print(df.isnull().sum())
    print(f"      ✓ All missing values handled")
else:
    print(f"      ✓ No missing values found")

# Encode target labels if categorical
print(f"\n   Step 3.4: Encode target labels")
target_col = config['target_column']
target_encoding = config['preprocessing']['target_encoding']

# Check if target is categorical
if df[target_col].dtype == 'object' or df[target_col].dtype == 'string':
    print(f"      Target is categorical, encoding...")
    df[target_col] = df[target_col].map(target_encoding)
    print(f"      Encoding: {target_encoding}")
    print(f"      ✓ Target encoded to binary: {df[target_col].unique()}")
elif set(df[target_col].unique()).issubset({'Yes', 'No'}):
    # Handle case where it's already stored as string but detected differently
    df[target_col] = df[target_col].map(target_encoding)
    print(f"      Encoding: {target_encoding}")
    print(f"      ✓ Target encoded to binary: {df[target_col].unique()}")
else:
    print(f"      Target is already numeric: {df[target_col].unique()}")

# Separate features and target
print(f"\n   Step 3.5: Separate features (X) and target (y)")
X = df.drop(config['target_column'], axis=1)
y = df[config['target_column']]

print(f"      Features (X) shape: {X.shape}")
print(f"      Target (y) shape: {y.shape}")
print(f"      ✓ Preprocessing complete")

# ============================================================
# STAGE 4: TRAIN-TEST SPLIT
# ============================================================
print("\n" + "=" * 80)
print("STAGE 4: TRAIN-TEST SPLIT")
print("-" * 80)

print(f"\n   Splitting data with parameters:")
print(f"      Test size: {config['test_size'] * 100}%")
print(f"      Random state: {config['random_state']} (for reproducibility)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config['test_size'],
    random_state=config['random_state']
)

print(f"\n   Split results:")
print(f"      Training set: {X_train.shape[0]} samples ({(1 - config['test_size']) * 100:.0f}%)")
print(f"      Testing set: {X_test.shape[0]} samples ({config['test_size'] * 100:.0f}%)")
print(f"      Training - Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")
print(f"      Testing - Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()}")
print(f"      ✓ Data split complete")

# ============================================================
# STAGE 5: MODEL TRAINING
# ============================================================
print("\n" + "=" * 80)
print("STAGE 5: MODEL TRAINING")
print("-" * 80)

print(f"\n   Initializing model: {config['model_type']}")

# Initialize model based on config
if config['model_type'] == 'logistic_regression':
    model = LogisticRegression(
        random_state=config['random_state'],
        max_iter=config['model_parameters']['max_iter']
    )
    print(f"      Model: Logistic Regression")
    print(f"      Parameters: max_iter={config['model_parameters']['max_iter']}, random_state={config['random_state']}")
else:
    print(f"   ✗ Error: Unknown model type '{config['model_type']}'")
    exit(1)

print(f"\n   Training model on {X_train.shape[0]} samples...")
model.fit(X_train, y_train)
print(f"      ✓ Model training complete")

# Display model coefficients
print(f"\n   Model coefficients:")
for i, col in enumerate(X.columns):
    print(f"      {col}: {model.coef_[0][i]:.4f}")
print(f"      Intercept: {model.intercept_[0]:.4f}")

# ============================================================
# STAGE 6: MODEL EVALUATION
# ============================================================
print("\n" + "=" * 80)
print("STAGE 6: MODEL EVALUATION")
print("-" * 80)

print(f"\n   Making predictions on test set ({X_test.shape[0]} samples)...")
y_pred = model.predict(X_test)

print(f"\n   Calculating evaluation metrics...")
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n   Evaluation Results:")
print(f"      Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"      F1-Score: {f1:.4f} ({f1 * 100:.2f}%)")

# Create metrics dictionary
metrics = {
    'accuracy': float(accuracy),
    'f1_score': float(f1),
    'test_samples': int(X_test.shape[0]),
    'train_samples': int(X_train.shape[0]),
    'model_type': config['model_type'],
    'random_state': config['random_state'],
    'test_size': config['test_size']
}

print(f"      ✓ Evaluation complete")

# ============================================================
# STAGE 7: SAVE ARTIFACTS
# ============================================================
print("\n" + "=" * 80)
print("STAGE 7: SAVE ARTIFACTS")
print("-" * 80)

# Save trained model
print(f"\n   Step 7.1: Save trained model")
model_path = config['output_model_path']
try:
    joblib.dump(model, model_path)
    print(f"      ✓ Model saved to: {model_path}")
except Exception as e:
    print(f"      ✗ Error saving model: {e}")

# Save metrics
print(f"\n   Step 7.2: Save evaluation metrics")
metrics_path = config['output_metrics_path']
try:
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"      ✓ Metrics saved to: {metrics_path}")
except Exception as e:
    print(f"      ✗ Error saving metrics: {e}")

# Display saved metrics
print(f"\n   Saved metrics content:")
print(json.dumps(metrics, indent=4))

# ============================================================
# PIPELINE SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)

print(f"""
   Pipeline Execution Complete!
   
   Stages Completed:
      ✓ Stage 1: Configuration loaded from {config_file}
      ✓ Stage 2: Data loaded from {config['dataset_path']} ({df.shape[0]} rows)
      ✓ Stage 3: Preprocessing (missing values handled, target encoded)
      ✓ Stage 4: Train-test split ({X_train.shape[0]} train, {X_test.shape[0]} test)
      ✓ Stage 5: Model trained ({config['model_type']})
      ✓ Stage 6: Model evaluated (Accuracy: {accuracy:.4f}, F1: {f1:.4f})
      ✓ Stage 7: Artifacts saved (model + metrics)
   
   Output Files:
      - Model: {model_path}
      - Metrics: {metrics_path}
   
   Reproducibility:
      - Random state: {config['random_state']}
      - All parameters defined in: {config_file}
      - Re-run this script to reproduce exact results
""")

print("=" * 80)
print("PIPELINE EXECUTION SUCCESSFUL")
print("=" * 80)
print()

# ============================================================
# MLOPS PRINCIPLES DEMONSTRATED
# ============================================================
print("MLOps Principles Demonstrated:")
print("   1. ✓ Configuration Management: All parameters in config.json")
print("   2. ✓ Reproducibility: Fixed random_state ensures same results")
print("   3. ✓ Logging: Clear stage-by-stage execution logs")
print("   4. ✓ Artifact Management: Model and metrics saved for reuse")
print("   5. ✓ Pipeline Structure: Load → Preprocess → Train → Evaluate → Save")
print("   6. ✓ Error Handling: Validation checks at each stage")
print("   7. ✓ Modularity: Each stage is independent and well-defined")
print()
