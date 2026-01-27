import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("EXPERIMENT 1: BASELINE MODEL vs CLEANED DATA")
print("=" * 60)
print()

# ============================================================
# A. BASELINE MODEL (Training with Raw Data)
# ============================================================
print("A. BASELINE MODEL (Training with Raw Data)")
print("-" * 60)

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

# Step 2: Identify input and output
print("\n2. Identifying input (X) and output (y)...")
X = df.drop('Purchased', axis=1)  # Features
y = df['Purchased']               # Target
print(f"   Features (X) shape: {X.shape}")
print(f"   Target (y) shape: {y.shape}")

# Step 3: Split the dataset
print("\n3. Splitting dataset into Training (70%) and Testing (30%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"   Training set size: {X_train.shape[0]}")
print(f"   Testing set size: {X_test.shape[0]}")

# Step 4: Train the baseline model
print("\n4. Training Baseline Model (Logistic Regression with raw data)...")
try:
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train, y_train)
    print("   Baseline model trained successfully!")

    print("\n5. Testing Baseline Model...")
    y_pred_baseline = baseline_model.predict(X_test)
    print(f"   Predictions: {y_pred_baseline}")
    
    print("\n6. Evaluating Baseline Model Performance...")
    baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
    baseline_f1 = f1_score(y_test, y_pred_baseline, zero_division=0)
    
    print(f"\n" + "=" * 60)
    print("BASELINE MODEL RESULTS (Raw Data)")
    print("=" * 60)
    print(f"Accuracy: {baseline_accuracy:.4f}")
    print(f"F1-Score: {baseline_f1:.4f}")
    print("=" * 60)
    
except ValueError as e:
    print(f"\n   ❌ ERROR: {str(e)}")
    print("\n" + "=" * 60)
    print("BASELINE MODEL RESULTS (Raw Data)")
    print("=" * 60)
    print("   ⚠️  Model Training FAILED due to missing values (NaN)")
    print("   Accuracy: Unable to compute")
    print("   F1-Score: Unable to compute")
    print("=" * 60)
    print("\n   📌 This demonstrates why DATA CLEANING is essential!")
    print("   Missing values must be handled before training ML models.")
    print("=" * 60)
