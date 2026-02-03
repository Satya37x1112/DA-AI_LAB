"""
EXPERIMENT 4: DATA QUALITY SCORECARD
=====================================
This script automatically assesses dataset quality using measurable metrics
and generates an overall data quality score.

Data Quality Dimensions:
1. Completeness - Missing value analysis
2. Uniqueness - Duplicate detection
3. Balance - Class distribution analysis
4. Privacy - PII detection
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EXPERIMENT 4: DATA QUALITY SCORECARD")
print("=" * 80)
print()

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("STEP 1: LOAD DATASET")
print("-" * 80)

dataset_file = 'raw_dataset.csv'
print(f"\n   Loading dataset from: {dataset_file}")

try:
    df = pd.read_csv(dataset_file)
    print(f"   ✓ Dataset loaded successfully")
except FileNotFoundError:
    print(f"   ✗ Error: Dataset file '{dataset_file}' not found!")
    exit(1)

print(f"\n   Dataset Information:")
print(f"      Shape: {df.shape}")
print(f"      Rows: {df.shape[0]}")
print(f"      Columns: {df.shape[1]}")
print(f"      Column names: {df.columns.tolist()}")

print(f"\n   First few rows:")
print(df.head())

print(f"\n   Data types:")
print(df.dtypes)

# ============================================================
# STEP 2: MISSING VALUE ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: MISSING VALUE ANALYSIS")
print("-" * 80)
print("\n   Why it matters: Missing values reduce data quality and can bias models")

# Calculate missing values per column
missing_per_column = df.isnull().sum()
missing_percentage_per_column = (df.isnull().sum() / len(df)) * 100

print(f"\n   Missing values per column:")
for col in df.columns:
    missing_count = missing_per_column[col]
    missing_pct = missing_percentage_per_column[col]
    status = "✓" if missing_count == 0 else "⚠"
    print(f"      {status} {col}: {missing_count} ({missing_pct:.2f}%)")

# Calculate overall missing value percentage
total_cells = df.shape[0] * df.shape[1]
total_missing = df.isnull().sum().sum()
overall_missing_pct = (total_missing / total_cells) * 100

print(f"\n   Overall Dataset:")
print(f"      Total cells: {total_cells}")
print(f"      Missing cells: {total_missing}")
print(f"      Missing percentage: {overall_missing_pct:.2f}%")

# Score for missing values (0-100, higher is better)
# 0% missing = 100 score, 100% missing = 0 score
missing_value_score = 100 - overall_missing_pct
print(f"\n   Completeness Score: {missing_value_score:.2f}/100")

# ============================================================
# STEP 3: DUPLICATE ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: DUPLICATE ANALYSIS")
print("-" * 80)
print("\n   Why it matters: Duplicates can lead to overfitting and biased evaluation")

# Identify duplicate rows
duplicate_rows = df.duplicated()
num_duplicates = duplicate_rows.sum()
duplicate_percentage = (num_duplicates / len(df)) * 100

print(f"\n   Duplicate Analysis:")
print(f"      Total rows: {len(df)}")
print(f"      Duplicate rows: {num_duplicates}")
print(f"      Duplicate percentage: {duplicate_percentage:.2f}%")

if num_duplicates > 0:
    print(f"      ⚠ Warning: Found {num_duplicates} duplicate rows")
    print(f"\n   First few duplicate rows:")
    print(df[duplicate_rows].head())
else:
    print(f"      ✓ No duplicate rows found")

# Score for duplicates (0-100, higher is better)
# 0% duplicates = 100 score, 100% duplicates = 0 score
duplicate_score = 100 - duplicate_percentage
print(f"\n   Uniqueness Score: {duplicate_score:.2f}/100")

# ============================================================
# STEP 4: CLASS IMBALANCE CHECK
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: CLASS IMBALANCE CHECK")
print("-" * 80)
print("\n   Why it matters: Severe imbalance can lead to biased models favoring majority class")

target_column = 'Purchased'

# Check if target column exists
if target_column not in df.columns:
    print(f"   ✗ Error: Target column '{target_column}' not found!")
    class_balance_score = 50  # Neutral score if column not found
else:
    # Analyze class distribution
    class_distribution = df[target_column].value_counts()
    class_percentages = (df[target_column].value_counts(normalize=True) * 100)
    
    print(f"\n   Target Column: {target_column}")
    print(f"   Class Distribution:")
    for class_label, count in class_distribution.items():
        pct = class_percentages[class_label]
        print(f"      Class '{class_label}': {count} ({pct:.2f}%)")
    
    # Calculate imbalance ratio
    if len(class_distribution) >= 2:
        majority_count = class_distribution.max()
        minority_count = class_distribution.min()
        imbalance_ratio = majority_count / minority_count
        
        # Calculate percentage split
        majority_pct = class_percentages.max()
        minority_pct = class_percentages.min()
        
        print(f"\n   Imbalance Analysis:")
        print(f"      Majority class: {majority_pct:.2f}%")
        print(f"      Minority class: {minority_pct:.2f}%")
        print(f"      Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Check if imbalanced (threshold: 70:30)
        imbalance_threshold = 70
        is_imbalanced = majority_pct > imbalance_threshold
        
        if is_imbalanced:
            print(f"      ⚠ Dataset is IMBALANCED (>{imbalance_threshold}:{100-imbalance_threshold} split)")
        else:
            print(f"      ✓ Dataset is relatively BALANCED")
        
        # Score for class balance (0-100, higher is better)
        # Perfect balance (50:50) = 100, severe imbalance = lower score
        # Use minority percentage as basis: 50% = perfect, 0% = worst
        class_balance_score = (minority_pct / 50) * 100
        if class_balance_score > 100:
            class_balance_score = 100
    else:
        print(f"      ⚠ Only one class found - cannot compute balance")
        class_balance_score = 0
    
    print(f"\n   Balance Score: {class_balance_score:.2f}/100")

# ============================================================
# STEP 5: BASIC PII DETECTION
# ============================================================
print("\n" + "=" * 80)
print("STEP 5: BASIC PII DETECTION")
print("-" * 80)
print("\n   Why it matters: PII poses privacy risks and may violate data protection regulations")

# Regular expressions for PII detection
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'

# Convert DataFrame to string representation for searching
string_columns = df.select_dtypes(include=['object']).columns

print(f"\n   Scanning {len(string_columns)} string columns for PII patterns...")

pii_found = {
    'emails': 0,
    'phones': 0,
    'total_rows_with_pii': 0
}

rows_with_pii = set()

# Scan each string column
for col in string_columns:
    col_emails = 0
    col_phones = 0
    
    for idx, value in df[col].items():
        if pd.isna(value):
            continue
        
        value_str = str(value)
        
        # Check for email patterns
        if re.search(email_pattern, value_str):
            col_emails += 1
            rows_with_pii.add(idx)
        
        # Check for phone patterns
        if re.search(phone_pattern, value_str):
            col_phones += 1
            rows_with_pii.add(idx)
    
    if col_emails > 0 or col_phones > 0:
        print(f"      Column '{col}': {col_emails} emails, {col_phones} phones")
        pii_found['emails'] += col_emails
        pii_found['phones'] += col_phones

pii_found['total_rows_with_pii'] = len(rows_with_pii)

print(f"\n   PII Detection Results:")
print(f"      Total emails found: {pii_found['emails']}")
print(f"      Total phone numbers found: {pii_found['phones']}")
print(f"      Rows with PII: {pii_found['total_rows_with_pii']} ({(pii_found['total_rows_with_pii']/len(df)*100):.2f}%)")

if pii_found['total_rows_with_pii'] > 0:
    print(f"      ⚠ Warning: PII detected in dataset - consider anonymization")
else:
    print(f"      ✓ No obvious PII patterns detected")

# Score for PII risk (0-100, higher is better = less PII)
# 0% PII = 100 score, 100% PII = 0 score
pii_risk_percentage = (pii_found['total_rows_with_pii'] / len(df)) * 100
pii_score = 100 - pii_risk_percentage
print(f"\n   Privacy Score: {pii_score:.2f}/100")

# ============================================================
# STEP 6: DATA QUALITY SCORING
# ============================================================
print("\n" + "=" * 80)
print("STEP 6: DATA QUALITY SCORING")
print("-" * 80)

print(f"\n   Individual Dimension Scores:")
print(f"      1. Completeness (Missing Values): {missing_value_score:.2f}/100")
print(f"      2. Uniqueness (Duplicates):       {duplicate_score:.2f}/100")
print(f"      3. Balance (Class Distribution):  {class_balance_score:.2f}/100")
print(f"      4. Privacy (PII Risk):            {pii_score:.2f}/100")

# Calculate weighted average (can adjust weights based on importance)
weights = {
    'completeness': 0.30,  # 30% weight
    'uniqueness': 0.25,    # 25% weight
    'balance': 0.25,       # 25% weight
    'privacy': 0.20        # 20% weight
}

final_quality_score = (
    missing_value_score * weights['completeness'] +
    duplicate_score * weights['uniqueness'] +
    class_balance_score * weights['balance'] +
    pii_score * weights['privacy']
)

print(f"\n   Scoring Weights:")
print(f"      Completeness: {weights['completeness']*100:.0f}%")
print(f"      Uniqueness:   {weights['uniqueness']*100:.0f}%")
print(f"      Balance:      {weights['balance']*100:.0f}%")
print(f"      Privacy:      {weights['privacy']*100:.0f}%")

print(f"\n   " + "=" * 50)
print(f"   FINAL DATA QUALITY SCORE: {final_quality_score:.2f}/100")
print(f"   " + "=" * 50)

# ============================================================
# STEP 7: DATASET RANKING
# ============================================================
print("\n" + "=" * 80)
print("STEP 7: DATASET RANKING")
print("-" * 80)

# Determine quality ranking
if final_quality_score >= 85:
    ranking = "EXCELLENT"
    emoji = "🌟"
    description = "High-quality dataset ready for production use"
elif final_quality_score >= 70:
    ranking = "GOOD"
    emoji = "✓"
    description = "Good quality dataset with minor issues"
elif final_quality_score >= 50:
    ranking = "FAIR"
    emoji = "⚠"
    description = "Fair quality dataset - consider improvements before use"
else:
    ranking = "POOR"
    emoji = "✗"
    description = "Poor quality dataset - significant improvements needed"

print(f"\n   Quality Ranking: {emoji} {ranking}")
print(f"   Description: {description}")

# Provide recommendations
print(f"\n   Recommendations:")
if missing_value_score < 90:
    print(f"      • Handle missing values (currently {overall_missing_pct:.2f}% missing)")
if duplicate_score < 100:
    print(f"      • Remove duplicate rows ({num_duplicates} duplicates found)")
if class_balance_score < 80:
    print(f"      • Address class imbalance using oversampling/undersampling")
if pii_score < 100:
    print(f"      • Anonymize or remove PII ({pii_found['total_rows_with_pii']} rows affected)")
if final_quality_score >= 85:
    print(f"      • ✓ Dataset is in good shape - no major issues detected")

# ============================================================
# STEP 8: DATA QUALITY SCORECARD (FINAL OUTPUT)
# ============================================================
print("\n" + "=" * 80)
print("DATA QUALITY SCORECARD - SUMMARY")
print("=" * 80)

scorecard = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                        DATA QUALITY SCORECARD                              ║
╚════════════════════════════════════════════════════════════════════════════╝

Dataset: {dataset_file}
Rows: {df.shape[0]} | Columns: {df.shape[1]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY DIMENSIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. COMPLETENESS (Missing Values)
   Score: {missing_value_score:.2f}/100
   Missing Cells: {total_missing}/{total_cells} ({overall_missing_pct:.2f}%)
   Status: {'✓ Excellent' if missing_value_score >= 90 else '⚠ Needs Attention' if missing_value_score >= 70 else '✗ Critical'}

2. UNIQUENESS (Duplicates)
   Score: {duplicate_score:.2f}/100
   Duplicate Rows: {num_duplicates}/{len(df)} ({duplicate_percentage:.2f}%)
   Status: {'✓ Excellent' if duplicate_score == 100 else '⚠ Needs Attention' if duplicate_score >= 90 else '✗ Critical'}

3. BALANCE (Class Distribution)
   Score: {class_balance_score:.2f}/100
   Class Split: {class_percentages.max():.1f}% / {class_percentages.min():.1f}%
   Status: {'✓ Balanced' if class_balance_score >= 80 else '⚠ Imbalanced' if class_balance_score >= 50 else '✗ Severely Imbalanced'}

4. PRIVACY (PII Risk)
   Score: {pii_score:.2f}/100
   Rows with PII: {pii_found['total_rows_with_pii']}/{len(df)} ({pii_risk_percentage:.2f}%)
   Status: {'✓ No PII Detected' if pii_score == 100 else '⚠ PII Found - Review Needed'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERALL ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Final Quality Score: {final_quality_score:.2f}/100
Quality Ranking: {ranking}
Assessment: {description}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

print(scorecard)

print("=" * 80)
print("SCORECARD GENERATION COMPLETE")
print("=" * 80)
print()

# Summary statistics table
print("Detailed Metrics Table:")
print("-" * 80)

metrics_table = pd.DataFrame({
    'Dimension': ['Completeness', 'Uniqueness', 'Balance', 'Privacy', 'OVERALL'],
    'Metric': [
        f'{overall_missing_pct:.2f}% Missing',
        f'{duplicate_percentage:.2f}% Duplicates',
        f'{class_percentages.min():.2f}% Minority Class',
        f'{pii_risk_percentage:.2f}% PII Rows',
        'Weighted Average'
    ],
    'Score': [
        f'{missing_value_score:.2f}/100',
        f'{duplicate_score:.2f}/100',
        f'{class_balance_score:.2f}/100',
        f'{pii_score:.2f}/100',
        f'{final_quality_score:.2f}/100'
    ],
    'Weight': [
        f'{weights["completeness"]*100:.0f}%',
        f'{weights["uniqueness"]*100:.0f}%',
        f'{weights["balance"]*100:.0f}%',
        f'{weights["privacy"]*100:.0f}%',
        '100%'
    ]
})

print(metrics_table.to_string(index=False))
print()

print("=" * 80)
print("Why Each Metric Matters:")
print("-" * 80)
print("""
1. COMPLETENESS (Missing Values):
   - Missing data reduces statistical power and can introduce bias
   - Imputation may not always recover true information
   - High missing rates may indicate data collection issues

2. UNIQUENESS (Duplicates):
   - Duplicates artificially inflate dataset size
   - Can cause data leakage in train-test splits
   - Lead to overfitting and overoptimistic performance metrics

3. BALANCE (Class Distribution):
   - Imbalanced datasets bias models toward majority class
   - Minority class may be underrepresented in predictions
   - Affects model evaluation and real-world performance

4. PRIVACY (PII Detection):
   - Personal information poses legal and ethical risks
   - May violate GDPR, HIPAA, or other privacy regulations
   - PII should be anonymized or removed before model training
""")

print("=" * 80)
