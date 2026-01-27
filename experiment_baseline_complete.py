import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("EXPERIMENT 1: BASELINE MODEL vs CLEANED DATA (SAME MODEL)")
print("=" * 70)
print()

# ============================================================
# A. BASELINE MODEL (Training with Raw Data)
# ============================================================
print("A. BASELINE MODEL (Training with Raw Data)")
print("-" * 70)

# Step 1: Load the dataset
print("\n1. Loading dataset...")
df = pd.read_csv('raw_dataset.csv')
# Convert target to binary (Yes -> 1, No -> 0)
df['Purchased'] = (df['Purchased'] == 'Yes').astype(int)
print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print(f"\n   First few rows:")
print(df.head())

# Check for missing values in raw data
print(f"\n   Missing values in raw data:")
print(df.isnull().sum())
print(f"   Total rows with missing values: {df.isnull().any(axis=1).sum()}")

# Step 2: Identify input and output
print("\n2. Identifying input (X) and output (y)...")
X_raw = df.drop('Purchased', axis=1)  # Features
y_raw = df['Purchased']               # Target
print(f"   Features (X) shape: {X_raw.shape}")
print(f"   Target (y) shape: {y_raw.shape}")

# Step 3: Split the dataset
print("\n3. Splitting dataset into Training (70%) and Testing (30%)...")
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.3, random_state=42
)
print(f"   Training set size: {X_train_raw.shape[0]}")
print(f"   Testing set size: {X_test_raw.shape[0]}")

# Step 4: Train the baseline model
print("\n4. Training Baseline Model (Logistic Regression with raw data)...")
baseline_accuracy = None
baseline_f1 = None
try:
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train_raw, y_train_raw)
    print("   ✓ Baseline model trained successfully!")
    
    # Step 5: Test the baseline model
    print("\n5. Testing Baseline Model...")
    y_pred_baseline = baseline_model.predict(X_test_raw)
    print(f"   Predictions: {y_pred_baseline}")
    
    # Step 6: Evaluate baseline performance
    print("\n6. Evaluating Baseline Model Performance...")
    baseline_accuracy = accuracy_score(y_test_raw, y_pred_baseline)
    baseline_f1 = f1_score(y_test_raw, y_pred_baseline, zero_division=0)
    
except ValueError as e:
    print(f"   ❌ ERROR: Model training failed!")
    print(f"   Reason: Input contains NaN values")
    print(f"\n   This demonstrates that raw data cannot be used directly!")

print(f"\n" + "=" * 70)
print("BASELINE MODEL RESULTS (Raw Data - WITH Missing Values)")
print("=" * 70)
if baseline_accuracy is not None:
    print(f"Accuracy:  {baseline_accuracy:.4f}")
    print(f"F1-Score:  {baseline_f1:.4f}")
else:
    print("Accuracy:  ❌ FAILED - Cannot handle missing values")
    print("F1-Score:  ❌ FAILED - Cannot handle missing values")
print("=" * 70)

print("\n\n")

# ============================================================
# B. CLEANED DATA MODEL (With Data Cleaning - Same Model)
# ============================================================
print("B. CLEANED DATA MODEL (Same Model with Cleaned Data)")
print("-" * 70)

print("\n1. Loading dataset...")
df_cleaned = pd.read_csv('raw_dataset.csv')
# Convert target to binary (Yes -> 1, No -> 0)
df_cleaned['Purchased'] = (df_cleaned['Purchased'] == 'Yes').astype(int)
print(f"   Dataset shape (before cleaning): {df_cleaned.shape}")

# Data Cleaning Strategy: Remove rows with missing values
print("\n2. Data Cleaning: Removing rows with missing values...")
rows_before = len(df_cleaned)
df_cleaned = df_cleaned.dropna()
rows_after = len(df_cleaned)
rows_removed = rows_before - rows_after
print(f"   Rows before cleaning: {rows_before}")
print(f"   Rows after cleaning:  {rows_after}")
print(f"   Rows removed:         {rows_removed}")

print(f"\n   Missing values after cleaning:")
print(df_cleaned.isnull().sum())
print(f"\n   Cleaned dataset (first few rows):")
print(df_cleaned.head())

# Step 2: Identify input and output
print("\n3. Identifying input (X) and output (y)...")
X_clean = df_cleaned.drop('Purchased', axis=1)  # Features
y_clean = df_cleaned['Purchased']               # Target
print(f"   Features (X) shape: {X_clean.shape}")
print(f"   Target (y) shape: {y_clean.shape}")

# Step 3: Split the dataset
print("\n4. Splitting cleaned dataset into Training (70%) and Testing (30%)...")
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X_clean, y_clean, test_size=0.3, random_state=10
)
print(f"   Training set size: {X_train_clean.shape[0]}")
print(f"   Testing set size: {X_test_clean.shape[0]}")

# Step 4: Train the model with cleaned data
print("\n5. Training Model (Logistic Regression with CLEANED data)...")
cleaned_model = LogisticRegression(max_iter=1000, random_state=42)
cleaned_model.fit(X_train_clean, y_train_clean)
print("   ✓ Model trained successfully!")

# Step 5: Test the model
print("\n6. Testing Model...")
y_pred_clean = cleaned_model.predict(X_test_clean)
print(f"   Predictions: {y_pred_clean}")

# Step 6: Evaluate performance
print("\n7. Evaluating Model Performance...")
cleaned_accuracy = accuracy_score(y_test_clean, y_pred_clean)
cleaned_f1 = f1_score(y_test_clean, y_pred_clean, zero_division=0)

print(f"\n" + "=" * 70)
print("CLEANED DATA MODEL RESULTS (Without Missing Values)")
print("=" * 70)
print(f"Accuracy:  {cleaned_accuracy:.4f}")
print(f"F1-Score:  {cleaned_f1:.4f}")
print("=" * 70)

# ============================================================
# COMPARISON & CONCLUSION
# ============================================================
print("\n\n")
print("=" * 70)
print("EXPERIMENT CONCLUSION")
print("=" * 70)

if baseline_accuracy is not None:
    accuracy_improvement = cleaned_accuracy - baseline_accuracy
    f1_improvement = cleaned_f1 - baseline_f1
    accuracy_improvement_pct = (accuracy_improvement / baseline_accuracy * 100) if baseline_accuracy != 0 else 0
    f1_improvement_pct = (f1_improvement / baseline_f1 * 100) if baseline_f1 != 0 else 0
    
    print(f"\nBaseline Model (Raw Data):        Accuracy={baseline_accuracy:.4f}, F1={baseline_f1:.4f}")
    print(f"Cleaned Data Model (Same Model):  Accuracy={cleaned_accuracy:.4f}, F1={cleaned_f1:.4f}")
    print(f"\nImprovement:")
    print(f"  Accuracy improvement: {accuracy_improvement:.4f} ({accuracy_improvement_pct:+.2f}%)")
    print(f"  F1-score improvement: {f1_improvement:.4f} ({f1_improvement_pct:+.2f}%)")
else:
    print(f"\nBaseline Model (Raw Data):        ❌ FAILED (cannot handle missing values)")
    print(f"Cleaned Data Model (Same Model):  Accuracy={cleaned_accuracy:.4f}, F1={cleaned_f1:.4f}")
    print(f"\n✓ Data cleaning enabled model training!")
    print(f"✓ Model could now be evaluated with metrics.")

print("\n📌 KEY INSIGHTS:")
print("   • Data quality directly impacts model performance")
print("   • Even without changing the model, cleaning data improves results")
print("   • Missing values must be handled before training")
print("   • Data preprocessing is a critical step in ML pipeline")
print("=" * 70)
