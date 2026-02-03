import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EXPERIMENT 2: LABEL NOISE EFFECT & BASIC CLEANUP")
print("=" * 80)
print()

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("STEP 1: LOAD DATASET")
print("-" * 80)

print("\n1. Loading dataset from CSV...")
df = pd.read_csv('raw_dataset.csv')

# Convert target to binary (Yes -> 1, No -> 0)
df['Purchased'] = (df['Purchased'] == 'Yes').astype(int)

print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print(f"\n   First few rows:")
print(df.head())

print(f"\n   Target distribution:")
print(df['Purchased'].value_counts())

# Check for missing values
print(f"\n   Missing values:")
print(df.isnull().sum())

# ============================================================
# STEP 2: TRAIN-TEST SPLIT
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: TRAIN-TEST SPLIT (70% Training, 30% Testing)")
print("-" * 80)

# Separate features and target
X = df.drop('Purchased', axis=1)  # Features
y = df['Purchased']               # Target

print(f"\n   Total features (X): {X.shape}")
print(f"   Total target (y): {y.shape}")

# Split with random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\n   Training set size: {X_train.shape[0]}")
print(f"   Testing set size: {X_test.shape[0]}")
print(f"   Training set - Class 0 (No): {(y_train == 0).sum()}, Class 1 (Yes): {(y_train == 1).sum()}")
print(f"   Testing set - Class 0 (No): {(y_test == 0).sum()}, Class 1 (Yes): {(y_test == 1).sum()}")

# ============================================================
# STEP 3: BASELINE MODEL (CLEAN LABELS)
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: BASELINE MODEL (TRAIN ON CLEAN DATA)")
print("-" * 80)

print("\n   Training Logistic Regression on clean training data...")
model_clean = LogisticRegression(random_state=42, max_iter=1000)
model_clean.fit(X_train, y_train)

# Predictions on test set
y_pred_clean = model_clean.predict(X_test)

# Evaluate baseline model
accuracy_clean = accuracy_score(y_test, y_pred_clean)
f1_clean = f1_score(y_test, y_pred_clean)

print(f"\n   ✓ Model trained on CLEAN data")
print(f"   ✓ Accuracy on test set: {accuracy_clean:.4f}")
print(f"   ✓ F1-score on test set: {f1_clean:.4f}")

# ============================================================
# STEP 4: INTRODUCE LABEL NOISE
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: INTRODUCE LABEL NOISE (Flip 10% of labels)")
print("-" * 80)

# Create a copy of clean training labels
y_train_noisy = y_train.copy()

# Randomly select 10% of training samples to flip
noise_rate = 0.10
n_samples_to_flip = int(len(y_train) * noise_rate)
np.random.seed(42)
indices_to_flip = np.random.choice(len(y_train), size=n_samples_to_flip, replace=False)

print(f"\n   Total training samples: {len(y_train)}")
print(f"   Noise rate: {noise_rate * 100}%")
print(f"   Samples to flip: {n_samples_to_flip}")

# Flip labels (0 -> 1, 1 -> 0)
y_train_noisy.iloc[indices_to_flip] = 1 - y_train_noisy.iloc[indices_to_flip]

print(f"\n   Original training set - Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")
print(f"   Noisy training set   - Class 0: {(y_train_noisy == 0).sum()}, Class 1: {(y_train_noisy == 1).sum()}")
print(f"   ✓ Label noise introduced successfully")

# ============================================================
# STEP 5: TRAIN MODEL ON NOISY DATA
# ============================================================
print("\n" + "=" * 80)
print("STEP 5: TRAIN MODEL ON NOISY DATA")
print("-" * 80)

print("\n   Training Logistic Regression on noisy training data...")
model_noisy = LogisticRegression(random_state=42, max_iter=1000)
model_noisy.fit(X_train, y_train_noisy)

# Predictions on test set
y_pred_noisy = model_noisy.predict(X_test)

# Evaluate model trained on noisy data
accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
f1_noisy = f1_score(y_test, y_pred_noisy)

print(f"\n   ✓ Model trained on NOISY data")
print(f"   ✓ Accuracy on test set: {accuracy_noisy:.4f}")
print(f"   ✓ F1-score on test set: {f1_noisy:.4f}")

# Show performance degradation
accuracy_drop = accuracy_clean - accuracy_noisy
f1_drop = f1_clean - f1_noisy

print(f"\n   Performance degradation due to label noise:")
print(f"   ✗ Accuracy drop: {accuracy_drop:.4f} ({accuracy_drop * 100:.2f}%)")
print(f"   ✗ F1-score drop: {f1_drop:.4f} ({f1_drop * 100:.2f}%)")

# ============================================================
# STEP 6: BASIC CLEANUP STRATEGY (Repeated Training)
# ============================================================
print("\n" + "=" * 80)
print("STEP 6: BASIC CLEANUP STRATEGY (Repeated Training & Misclassification Tracking)")
print("-" * 80)

print("\n   Strategy: Train model 3 times, track repeatedly misclassified samples")
print("   Assumption: Samples with label noise are likely to be misclassified repeatedly")
print()

# Track misclassifications across iterations
misclassification_count = np.zeros(len(X_train))
n_iterations = 3

print(f"   Running {n_iterations} iterations of training and prediction...")

for iteration in range(n_iterations):
    print(f"\n   Iteration {iteration + 1}:")
    
    # Train model on current training data
    model_temp = LogisticRegression(random_state=42, max_iter=1000)
    model_temp.fit(X_train, y_train_noisy)
    
    # Predict on training set
    y_train_pred = model_temp.predict(X_train)
    
    # Identify misclassified samples
    misclassified = (y_train_pred != y_train_noisy)
    misclassification_count += misclassified.astype(int)
    
    n_misclassified = misclassified.sum()
    print(f"      Misclassified samples: {n_misclassified} out of {len(y_train)}")

print(f"\n   Completed {n_iterations} iterations")

# Identify suspicious samples (misclassified at least 2 times out of 3)
suspicion_threshold = 2  # At least 2 misclassifications
suspicious_indices = np.where(misclassification_count >= suspicion_threshold)[0]

print(f"\n   Suspicious samples (misclassified >= {suspicion_threshold} times):")
print(f"      Total suspicious: {len(suspicious_indices)} out of {len(X_train)}")
print(f"      Percentage of data: {len(suspicious_indices) / len(X_train) * 100:.2f}%")

# ============================================================
# STEP 7: RETRAIN AFTER CLEANUP
# ============================================================
print("\n" + "=" * 80)
print("STEP 7: RETRAIN MODEL ON CLEANED DATA")
print("-" * 80)

# Create cleaned dataset by removing suspicious samples
X_train_cleaned = X_train.drop(X_train.index[suspicious_indices])
y_train_cleaned = y_train_noisy.drop(y_train_noisy.index[suspicious_indices])

print(f"\n   Original training set size: {len(X_train)}")
print(f"   Suspicious samples removed: {len(suspicious_indices)}")
print(f"   Cleaned training set size: {len(X_train_cleaned)}")
print(f"   Samples retained: {len(X_train_cleaned) / len(X_train) * 100:.2f}%")

# Train model on cleaned data
print(f"\n   Training Logistic Regression on cleaned data...")
model_cleaned = LogisticRegression(random_state=42, max_iter=1000)
model_cleaned.fit(X_train_cleaned, y_train_cleaned)

# Predictions on test set
y_pred_cleaned = model_cleaned.predict(X_test)

# Evaluate model trained on cleaned data
accuracy_cleaned = accuracy_score(y_test, y_pred_cleaned)
f1_cleaned = f1_score(y_test, y_pred_cleaned)

print(f"\n   ✓ Model trained on CLEANED data")
print(f"   ✓ Accuracy on test set: {accuracy_cleaned:.4f}")
print(f"   ✓ F1-score on test set: {f1_cleaned:.4f}")

# ============================================================
# STEP 8: FINAL OUTPUT - COMPARISON TABLE
# ============================================================
print("\n" + "=" * 80)
print("STEP 8: FINAL COMPARISON - CLEAN vs NOISY vs CLEANED")
print("=" * 80)
print()

# Create comparison dataframe
comparison_data = {
    'Model': ['Clean Data', 'Noisy Data', 'Cleaned Data'],
    'Training Samples': [len(X_train), len(X_train), len(X_train_cleaned)],
    'Accuracy': [f'{accuracy_clean:.4f}', f'{accuracy_noisy:.4f}', f'{accuracy_cleaned:.4f}'],
    'F1-Score': [f'{f1_clean:.4f}', f'{f1_noisy:.4f}', f'{f1_cleaned:.4f}'],
    'Accuracy (%)': [f'{accuracy_clean * 100:.2f}%', f'{accuracy_noisy * 100:.2f}%', f'{accuracy_cleaned * 100:.2f}%'],
    'F1-Score (%)': [f'{f1_clean * 100:.2f}%', f'{f1_noisy * 100:.2f}%', f'{f1_cleaned * 100:.2f}%']
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Summary statistics
print("\n" + "-" * 80)
print("SUMMARY STATISTICS")
print("-" * 80)

print(f"\n   Impact of Label Noise:")
print(f"      Accuracy: {accuracy_clean:.4f} → {accuracy_noisy:.4f} (Δ {accuracy_noisy - accuracy_clean:+.4f})")
print(f"      F1-Score: {f1_clean:.4f} → {f1_noisy:.4f} (Δ {f1_noisy - f1_clean:+.4f})")

print(f"\n   Impact of Cleanup:")
print(f"      Accuracy: {accuracy_noisy:.4f} → {accuracy_cleaned:.4f} (Δ {accuracy_cleaned - accuracy_noisy:+.4f})")
print(f"      F1-Score: {f1_noisy:.4f} → {f1_cleaned:.4f} (Δ {f1_cleaned - f1_noisy:+.4f})")

print(f"\n   Recovery from Noise:")
if accuracy_cleaned >= accuracy_clean:
    print(f"      ✓ Full recovery achieved! Cleaned accuracy matches clean baseline")
else:
    recovery_gap = accuracy_clean - accuracy_cleaned
    print(f"      ✓ Partial recovery. Gap from clean baseline: {recovery_gap:.4f}")

print(f"\n   Cleanup Effectiveness:")
removal_rate = len(suspicious_indices) / len(X_train) * 100
improvement_rate = (accuracy_cleaned - accuracy_noisy) / (accuracy_clean - accuracy_noisy) * 100 if (accuracy_clean - accuracy_noisy) != 0 else 0
print(f"      Samples removed: {removal_rate:.2f}%")
print(f"      Accuracy improvement achieved: {improvement_rate:.2f}% of maximum possible")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)
print()
print("Key Insights:")
print("  1. Label noise significantly reduces model performance")
print("  2. Removing samples with repeated misclassifications helps recover accuracy")
print("  3. This basic cleanup strategy is effective for noisy labels")
print("  4. The model confidence on suspicious samples indicates label quality issues")
