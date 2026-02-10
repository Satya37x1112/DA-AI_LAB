"""
EXPERIMENT 9: ONE-FILE COMPLETE EDA PROGRAM (Step-by-step)
=============================================================
Objective: Perform end-to-end Exploratory Data Analysis

This program demonstrates:
1) Define Objective
2) Load and Inspect Data
3) Data Cleaning
4) Univariate Analysis
5) Bivariate / Multivariate Analysis
6) Outlier Detection
7) Feature Transformation / Engineering

Example Objective: "Does income depend on age?"

How to use:
1) If you have a CSV, set USE_CSV=True and provide CSV_PATH.
2) Otherwise it runs on a small sample dataset.

Author: Data Engineering & Analytics Lab
Date: February 10, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eda_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# -------------------------------
# 1) Define Objective
# -------------------------------
OBJECTIVE = "Does income depend on age?"
print("\n" + "="*60)
print(f"OBJECTIVE: {OBJECTIVE}")
print("="*60 + "\n")
logger.info(f"Starting EDA with objective: {OBJECTIVE}")

# -------------------------------
# 2) Load and Inspect Data
# -------------------------------
USE_CSV = False
CSV_PATH = "data.csv"  # change if USE_CSV=True

logger.info("Loading data...")
if USE_CSV:
    try:
        df = pd.read_csv(CSV_PATH)
        logger.info(f"Loaded CSV from: {CSV_PATH}")
    except FileNotFoundError:
        logger.error(f"CSV file not found: {CSV_PATH}")
        raise
else:
    # Sample dataset
    df = pd.DataFrame({
        "age":    [22, 25, 29, 32, 35, 40, 45, 50, 28, 33, 38, 60, 27, np.nan, 48],
        "income": [18000, 22000, 28000, 32000, 40000, 52000, 60000, 75000, 30000, 35000, 48000, 120000, 26000, 45000, np.nan],
        "city":   ["Delhi","Delhi","Pune","Mumbai","Delhi","Kolkata","Bangalore","Bangalore","Mumbai","Pune","Delhi","Mumbai","Kolkata","Delhi","Pune"],
        "gender": ["M","M","F","F","M","F","M","M","F","M","F","M","F","M","M"]
    })
    logger.info("Using sample dataset (15 records)")

print("\n" + "─"*60)
print("SECTION 1: RAW DATA INSPECTION")
print("─"*60)
print(f"\n✓ Shape (rows, cols): {df.shape}")
logger.info(f"Dataset shape: {df.shape}")

print("\n📊 First 5 rows:")
print(df.head())

print("\n📋 Dataset Info:")
print(df.info())

print("\n❓ Missing values count:")
missing_counts = df.isna().sum()
print(missing_counts)
logger.info(f"Missing values: {missing_counts.to_dict()}")

print("\n📈 Descriptive Statistics (numeric + categorical):")
print(df.describe(include="all"))

# -------------------------------
# 3) Data Cleaning
# -------------------------------
print("\n" + "─"*60)
print("SECTION 2: DATA CLEANING")
print("─"*60)
logger.info("Starting data cleaning...")

# Remove duplicates
before = len(df)
df = df.drop_duplicates()
after = len(df)
duplicates_removed = before - after
print(f"\n✓ Duplicates removed: {duplicates_removed}")
logger.info(f"Duplicates removed: {duplicates_removed}")

# Fix incorrect data formats (coerce invalid to NaN)
for col in ["age", "income"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.info(f"Converted '{col}' to numeric")

# Handle missing values (drop rows missing in key columns)
required_cols = ["age", "income"]
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    error_msg = f"Your dataset is missing required columns: {missing_required}"
    logger.error(error_msg)
    raise ValueError(error_msg)

df_clean = df.dropna(subset=required_cols).copy()
print(f"✓ After dropping missing age/income: {df_clean.shape}")
logger.info(f"Dataset after cleaning: {df_clean.shape}")

print("\n✓ Remaining missing values:")
print(df_clean.isna().sum())

# -------------------------------
# 4) Univariate Analysis
# -------------------------------
print("\n" + "─"*60)
print("SECTION 3: UNIVARIATE ANALYSIS")
print("─"*60)
logger.info("Performing univariate analysis...")

age_mean = df_clean["age"].mean()
age_median = df_clean["age"].median()
income_mean = df_clean["income"].mean()
income_median = df_clean["income"].median()

print(f"\n📊 Age Statistics:")
print(f"   Mean: {age_mean:.2f}")
print(f"   Median: {age_median:.2f}")
print(f"   Min: {df_clean['age'].min():.2f}")
print(f"   Max: {df_clean['age'].max():.2f}")

print(f"\n💰 Income Statistics:")
print(f"   Mean: ₹{income_mean:,.2f}")
print(f"   Median: ₹{income_median:,.2f}")
print(f"   Min: ₹{df_clean['income'].min():,.2f}")
print(f"   Max: ₹{df_clean['income'].max():,.2f}")

if "city" in df_clean.columns:
    print("\n🏙️ City Distribution:")
    city_counts = df_clean["city"].value_counts()
    print(city_counts)
    logger.info(f"City distribution: {city_counts.to_dict()}")

if "gender" in df_clean.columns:
    print("\n👥 Gender Distribution:")
    gender_counts = df_clean["gender"].value_counts()
    print(gender_counts)
    logger.info(f"Gender distribution: {gender_counts.to_dict()}")

# Visualization 1: Income Distribution
print("\n📊 Generating visualization 1: Income Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_clean["income"], bins=10, color='skyblue', edgecolor='black')
axes[0].set_title("Income Distribution (Histogram)", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Income (₹)", fontsize=12)
axes[0].set_ylabel("Frequency", fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(df_clean["income"], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
axes[1].set_title("Income Distribution (Boxplot)", fontsize=14, fontweight='bold')
axes[1].set_ylabel("Income (₹)", fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_income_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("Saved: eda_income_distribution.png")
print("   ✓ Saved: eda_income_distribution.png")

# Visualization 2: Age Distribution
print("\n📊 Generating visualization 2: Age Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_clean["age"], bins=10, color='lightgreen', edgecolor='black')
axes[0].set_title("Age Distribution (Histogram)", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Age (years)", fontsize=12)
axes[0].set_ylabel("Frequency", fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(df_clean["age"], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightcoral'))
axes[1].set_title("Age Distribution (Boxplot)", fontsize=14, fontweight='bold')
axes[1].set_ylabel("Age (years)", fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("Saved: eda_age_distribution.png")
print("   ✓ Saved: eda_age_distribution.png")

# -------------------------------
# 5) Bivariate / Multivariate Analysis
# -------------------------------
print("\n" + "─"*60)
print("SECTION 4: BIVARIATE / MULTIVARIATE ANALYSIS")
print("─"*60)
logger.info("Performing bivariate/multivariate analysis...")

# Visualization 3: Scatter plot
print("\n📊 Generating visualization 3: Age vs Income Scatter Plot...")
plt.figure(figsize=(10, 6))
plt.scatter(df_clean["age"], df_clean["income"], alpha=0.7, s=100, 
            c='steelblue', edgecolors='black', linewidth=0.5)
plt.title("Age vs Income (Scatter Plot)", fontsize=16, fontweight='bold')
plt.xlabel("Age (years)", fontsize=12)
plt.ylabel("Income (₹)", fontsize=12)
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df_clean["age"], df_clean["income"], 1)
p = np.poly1d(z)
plt.plot(df_clean["age"], p(df_clean["age"]), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
plt.legend()

plt.tight_layout()
plt.savefig('eda_age_vs_income_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("Saved: eda_age_vs_income_scatter.png")
print("   ✓ Saved: eda_age_vs_income_scatter.png")

# Correlation analysis
corr = df_clean[["age", "income"]].corr()
print("\n📊 Correlation Matrix (Age vs Income):")
print(corr)
logger.info(f"Correlation (age vs income): {corr.loc['age', 'income']:.4f}")

# Visualization 4: Correlation heatmap
print("\n📊 Generating visualization 4: Correlation Heatmap...")
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f', 
            square=True, linewidths=2, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap: Age vs Income", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("Saved: eda_correlation_heatmap.png")
print("   ✓ Saved: eda_correlation_heatmap.png")

# Group-wise analysis
if "city" in df_clean.columns:
    print("\n🏙️ Income Statistics by City:")
    city_stats = df_clean.groupby("city")["income"].agg(
        ["count", "mean", "median", "min", "max"]
    ).sort_values("mean", ascending=False)
    print(city_stats)
    logger.info(f"City-wise income stats:\n{city_stats}")
    
    # Visualization 5: Income by City
    print("\n📊 Generating visualization 5: Income by City...")
    plt.figure(figsize=(10, 6))
    city_means = df_clean.groupby("city")["income"].mean().sort_values(ascending=False)
    city_means.plot(kind='bar', color='teal', edgecolor='black')
    plt.title("Average Income by City", fontsize=14, fontweight='bold')
    plt.xlabel("City", fontsize=12)
    plt.ylabel("Average Income (₹)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('eda_income_by_city.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: eda_income_by_city.png")
    print("   ✓ Saved: eda_income_by_city.png")

if "gender" in df_clean.columns:
    print("\n👥 Income Statistics by Gender:")
    gender_stats = df_clean.groupby("gender")["income"].agg(
        ["count", "mean", "median", "min", "max"]
    ).sort_values("mean", ascending=False)
    print(gender_stats)
    logger.info(f"Gender-wise income stats:\n{gender_stats}")
    
    # Visualization 6: Income by Gender
    print("\n📊 Generating visualization 6: Income by Gender...")
    plt.figure(figsize=(8, 6))
    df_clean.boxplot(column='income', by='gender', patch_artist=True, figsize=(8, 6))
    plt.title("Income Distribution by Gender", fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remove default title
    plt.xlabel("Gender", fontsize=12)
    plt.ylabel("Income (₹)", fontsize=12)
    plt.tight_layout()
    plt.savefig('eda_income_by_gender.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: eda_income_by_gender.png")
    print("   ✓ Saved: eda_income_by_gender.png")

# -------------------------------
# 6) Outlier Detection
# -------------------------------
print("\n" + "─"*60)
print("SECTION 5: OUTLIER DETECTION")
print("─"*60)
logger.info("Performing outlier detection...")

# IQR method
q1 = df_clean["income"].quantile(0.25)
q3 = df_clean["income"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers_iqr = df_clean[(df_clean["income"] < lower) | (df_clean["income"] > upper)]
print(f"\n📊 IQR Method:")
print(f"   Q1 (25th percentile): ₹{q1:,.2f}")
print(f"   Q3 (75th percentile): ₹{q3:,.2f}")
print(f"   IQR: ₹{iqr:,.2f}")
print(f"   Lower bound: ₹{lower:,.2f}")
print(f"   Upper bound: ₹{upper:,.2f}")
print(f"   Outliers detected: {len(outliers_iqr)}")
logger.info(f"IQR outliers: {len(outliers_iqr)}")

if len(outliers_iqr) > 0:
    print("\n   Outlier records (IQR method):")
    print(outliers_iqr[["age", "income"]].sort_values("income"))

# Z-score method
mean_income = df_clean["income"].mean()
std_income = df_clean["income"].std()
df_clean["z_income"] = (df_clean["income"] - mean_income) / std_income

outliers_z = df_clean[df_clean["z_income"].abs() > 3]
print(f"\n📊 Z-Score Method (threshold > 3):")
print(f"   Mean income: ₹{mean_income:,.2f}")
print(f"   Std deviation: ₹{std_income:,.2f}")
print(f"   Outliers detected: {len(outliers_z)}")
logger.info(f"Z-score outliers: {len(outliers_z)}")

if len(outliers_z) > 0:
    print("\n   Outlier records (Z-score method):")
    print(outliers_z[["age", "income", "z_income"]].sort_values("z_income"))

# Visualization 7: Outlier detection
print("\n📊 Generating visualization 7: Outlier Detection Methods...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# IQR visualization
axes[0].boxplot(df_clean["income"], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
axes[0].axhline(y=lower, color='r', linestyle='--', label=f'Lower: {lower:.0f}')
axes[0].axhline(y=upper, color='r', linestyle='--', label=f'Upper: {upper:.0f}')
axes[0].set_title("Outlier Detection (IQR Method)", fontsize=14, fontweight='bold')
axes[0].set_ylabel("Income (₹)", fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Z-score visualization
axes[1].scatter(df_clean["age"], df_clean["z_income"], alpha=0.7, s=100, 
                c='steelblue', edgecolors='black', linewidth=0.5)
axes[1].axhline(y=3, color='r', linestyle='--', label='Z = +3')
axes[1].axhline(y=-3, color='r', linestyle='--', label='Z = -3')
axes[1].set_title("Outlier Detection (Z-Score Method)", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Age (years)", fontsize=12)
axes[1].set_ylabel("Z-Score (Income)", fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_outlier_detection.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("Saved: eda_outlier_detection.png")
print("   ✓ Saved: eda_outlier_detection.png")

# -------------------------------
# 7) Feature Transformation / Engineering
# -------------------------------
print("\n" + "─"*60)
print("SECTION 6: FEATURE TRANSFORMATION / ENGINEERING")
print("─"*60)
logger.info("Performing feature engineering...")

# Standardization (simple, no sklearn)
df_clean["age_std"] = (df_clean["age"] - df_clean["age"].mean()) / df_clean["age"].std()
df_clean["income_std"] = (df_clean["income"] - df_clean["income"].mean()) / df_clean["income"].std()
print("\n✓ Created standardized features: age_std, income_std")
logger.info("Created standardized features")

# One-hot encoding for categorical columns if present
cat_cols = [c for c in ["city", "gender"] if c in df_clean.columns]
df_final = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
print(f"✓ Applied one-hot encoding to: {cat_cols}")
logger.info(f"One-hot encoded columns: {cat_cols}")

# Create a new feature
df_final["income_per_age"] = df_final["income"] / df_final["age"]
print("✓ Created new feature: income_per_age")
logger.info("Created derived feature: income_per_age")

print(f"\n📊 Final dataset shape: {df_final.shape}")
print("\n📋 Final dataset columns:")
print(list(df_final.columns))

print("\n📋 Final dataset (first 5 rows):")
print(df_final.head())

# Save final processed dataset
output_file = "eda_processed_data.csv"
df_final.to_csv(output_file, index=False)
print(f"\n✓ Saved processed dataset: {output_file}")
logger.info(f"Saved processed dataset: {output_file}")

# -------------------------------
# Summary Output
# -------------------------------
print("\n" + "="*60)
print("EDA SUMMARY REPORT")
print("="*60)

print(f"\n📊 OBJECTIVE: {OBJECTIVE}")
print(f"\n📈 DATASET OVERVIEW:")
print(f"   Original records: {before}")
print(f"   Duplicates removed: {duplicates_removed}")
print(f"   Final records: {len(df_final)}")
print(f"   Total features: {df_final.shape[1]}")

print(f"\n📊 KEY STATISTICS:")
print(f"   Average Age: {age_mean:.2f} years")
print(f"   Average Income: ₹{income_mean:,.2f}")
print(f"   Correlation (age vs income): {corr.loc['age', 'income']:.4f}")

print(f"\n🎯 CORRELATION INTERPRETATION:")
corr_value = corr.loc['age', 'income']
if abs(corr_value) >= 0.7:
    strength = "STRONG"
elif abs(corr_value) >= 0.4:
    strength = "MODERATE"
else:
    strength = "WEAK"

direction = "POSITIVE" if corr_value > 0 else "NEGATIVE"
print(f"   {strength} {direction} correlation detected")
print(f"   Income {'INCREASES' if corr_value > 0 else 'DECREASES'} with age")

print(f"\n⚠️ OUTLIERS DETECTED:")
print(f"   IQR method: {len(outliers_iqr)} outliers")
print(f"   Z-score method: {len(outliers_z)} outliers")

print(f"\n📁 OUTPUT FILES GENERATED:")
print(f"   1. eda_income_distribution.png")
print(f"   2. eda_age_distribution.png")
print(f"   3. eda_age_vs_income_scatter.png")
print(f"   4. eda_correlation_heatmap.png")
if "city" in df_clean.columns:
    print(f"   5. eda_income_by_city.png")
if "gender" in df_clean.columns:
    print(f"   6. eda_income_by_gender.png")
print(f"   7. eda_outlier_detection.png")
print(f"   8. eda_processed_data.csv")
print(f"   9. eda_analysis.log")

print("\n" + "="*60)
print("✅ EDA COMPLETE - All analyses finished successfully!")
print("="*60 + "\n")
logger.info("EDA analysis completed successfully")

# Create summary statistics file
summary = {
    "objective": OBJECTIVE,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": {
        "original_records": before,
        "duplicates_removed": duplicates_removed,
        "final_records": len(df_final),
        "total_features": df_final.shape[1]
    },
    "statistics": {
        "age_mean": float(age_mean),
        "age_median": float(age_median),
        "income_mean": float(income_mean),
        "income_median": float(income_median),
        "correlation_age_income": float(corr.loc['age', 'income'])
    },
    "outliers": {
        "iqr_method": len(outliers_iqr),
        "zscore_method": len(outliers_z)
    }
}

import json
with open('eda_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)
print("✓ Saved summary report: eda_summary.json\n")
logger.info("Saved summary JSON file")
