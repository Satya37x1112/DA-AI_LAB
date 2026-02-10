"""
Data Wrangling Toolkit - Core 8 Steps
======================================
Experiment Title: Data Wrangling Toolkit (Core 8 Steps)

Aim: Practice and implement common data wrangling operations used in 
     real-world data preprocessing.

Author: Data Analytics Lab
Date: February 10, 2026

This program demonstrates 8 essential data preprocessing techniques:
1. Handle Missing Values
2. Encode Categorical Data
3. Feature Scaling
4. Outlier Detection & Removal
5. GroupBy Operations
6. Pivot & Melt
7. Remove Duplicates
8. Train-Test Split
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_FILE = 'employees.csv'
OUTPUT_FILE = 'employees_cleaned.csv'


# ============================================================================
# STEP 0: LOAD DATA
# ============================================================================

def load_data(filepath=CSV_FILE):
    """
    Load dataset from CSV file with error handling.
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If dataset is empty
    """
    print("\n" + "=" * 70)
    print("STEP 0: LOADING DATA".center(70))
    print("=" * 70)
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File '{filepath}' not found!")
        
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Check if dataset is empty
        if df.empty:
            raise ValueError("Dataset is empty!")
        
        print(f"✓ Data loaded successfully from '{filepath}'")
        print(f"✓ Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"✓ Columns: {list(df.columns)}")
        
        print("\n--- First 5 Rows ---")
        print(df.head())
        
        print("\n--- Dataset Info ---")
        print(df.info())
        
        print("\n--- Statistical Summary ---")
        print(df.describe())
        
        return df
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        raise
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise


# ============================================================================
# STEP 1: HANDLE MISSING VALUES
# ============================================================================

def handle_missing(df, threshold=0.5):
    """
    Handle missing values in the dataset.
    
    Strategy:
    - Numerical columns: Fill with median
    - Categorical columns: Fill with mode
    - Drop rows if missing values exceed threshold
    
    Args:
        df (pd.DataFrame): Input dataframe
        threshold (float): Threshold for dropping rows (0 to 1)
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    print("\n" + "=" * 70)
    print("STEP 1: HANDLE MISSING VALUES".center(70))
    print("=" * 70)
    
    df_clean = df.copy()
    
    # Detect missing values
    missing_count = df_clean.isnull().sum()
    missing_percentage = (missing_count / len(df_clean)) * 100
    
    print("\n--- Missing Values Detection ---")
    missing_df = pd.DataFrame({
        'Column': missing_count.index,
        'Missing_Count': missing_count.values,
        'Missing_Percentage': missing_percentage.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("No missing values found.")
    
    # Drop rows where missing values exceed threshold
    missing_ratio = df_clean.isnull().sum(axis=1) / df_clean.shape[1]
    rows_to_drop = missing_ratio > threshold
    if rows_to_drop.sum() > 0:
        print(f"\n✓ Dropping {rows_to_drop.sum()} rows with > {threshold*100}% missing values")
        df_clean = df_clean[~rows_to_drop]
    
    # Fill missing values in numerical columns with median
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"✓ Filled '{col}' with median: {median_val:.2f}")
    
    # Fill missing values in categorical columns with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"✓ Filled '{col}' with mode: {mode_val}")
    
    print(f"\n✓ Final shape after handling missing values: {df_clean.shape}")
    print("\n" + "=" * 70)
    print("✓ STEP 1 COMPLETED — MISSING VALUES HANDLED".center(70))
    print("=" * 70)
    
    return df_clean


# ============================================================================
# STEP 2: ENCODING CATEGORICAL DATA
# ============================================================================

def encode_data(df):
    """
    Encode categorical variables for machine learning.
    
    - Label Encoding: Binary categorical (Gender)
    - One-Hot Encoding: Multi-category variables (Department, City)
    
    Why encoding?
    Machine learning models require numerical input. Categorical data
    must be converted to numbers while preserving information.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical variables
    """
    print("\n" + "=" * 70)
    print("STEP 2: ENCODING CATEGORICAL DATA".center(70))
    print("=" * 70)
    
    df_encoded = df.copy()
    
    print("\n--- Why Encoding is Required ---")
    print("Machine learning algorithms work with numerical data.")
    print("Categorical variables must be converted to numbers.")
    print("• Label Encoding: For binary/ordinal categories (0, 1, 2...)")
    print("• One-Hot Encoding: For nominal categories (creates dummy variables)")
    
    # Label Encoding for Gender (binary categorical)
    if 'Gender' in df_encoded.columns:
        le = LabelEncoder()
        original_values = df_encoded['Gender'].unique()
        df_encoded['Gender_Encoded'] = le.fit_transform(df_encoded['Gender'])
        print(f"\n✓ Label Encoding applied on 'Gender'")
        print(f"  Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # One-Hot Encoding for Department and City
    columns_to_encode = []
    if 'Department' in df_encoded.columns:
        columns_to_encode.append('Department')
    if 'City' in df_encoded.columns:
        columns_to_encode.append('City')
    
    if columns_to_encode:
        df_encoded = pd.get_dummies(df_encoded, columns=columns_to_encode, prefix=columns_to_encode)
        print(f"\n✓ One-Hot Encoding applied on: {columns_to_encode}")
        print(f"  New columns created: {[col for col in df_encoded.columns if any(prefix in col for prefix in columns_to_encode)]}")
    
    print(f"\n✓ Shape after encoding: {df_encoded.shape}")
    print("\n--- Sample of Encoded Data ---")
    print(df_encoded.head(3))
    
    print("\n" + "=" * 70)
    print("✓ STEP 2 COMPLETED — ENCODING DONE".center(70))
    print("=" * 70)
    
    return df_encoded


# ============================================================================
# STEP 3: FEATURE SCALING
# ============================================================================

def scale_features(df, columns_to_scale=['Salary', 'Experience_Years']):
    """
    Apply feature scaling techniques.
    
    - Min-Max Scaling: Scales data to [0, 1] range
    - Standardization: Scales data to mean=0, std=1 (Z-score)
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns_to_scale (list): Columns to scale
        
    Returns:
        pd.DataFrame: Dataframe with scaled features
    """
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE SCALING".center(70))
    print("=" * 70)
    
    df_scaled = df.copy()
    
    # Validate columns exist
    valid_columns = [col for col in columns_to_scale if col in df_scaled.columns]
    
    if not valid_columns:
        print("⚠ No valid columns found for scaling. Skipping...")
        return df_scaled
    
    print(f"\n--- Columns to scale: {valid_columns} ---")
    
    # Show original statistics
    print("\n--- Before Scaling ---")
    print(df_scaled[valid_columns].describe())
    
    # Min-Max Scaling
    scaler_minmax = MinMaxScaler()
    for col in valid_columns:
        df_scaled[f'{col}_MinMax'] = scaler_minmax.fit_transform(df_scaled[[col]])
    print(f"\n✓ Min-Max Scaling applied (range: 0 to 1)")
    
    # Standardization (Z-score)
    scaler_standard = StandardScaler()
    for col in valid_columns:
        df_scaled[f'{col}_Standard'] = scaler_standard.fit_transform(df_scaled[[col]])
    print(f"✓ Standardization applied (mean: 0, std: 1)")
    
    # Show scaled statistics
    scaled_cols = [f'{col}_MinMax' for col in valid_columns] + [f'{col}_Standard' for col in valid_columns]
    print("\n--- After Scaling ---")
    print(df_scaled[scaled_cols].describe())
    
    print("\n--- Comparison Example (First 5 rows) ---")
    comparison_cols = []
    for col in valid_columns:
        comparison_cols.extend([col, f'{col}_MinMax', f'{col}_Standard'])
    print(df_scaled[comparison_cols].head())
    
    print("\n" + "=" * 70)
    print("✓ STEP 3 COMPLETED — FEATURE SCALING DONE".center(70))
    print("=" * 70)
    
    return df_scaled


# ============================================================================
# STEP 4: OUTLIER DETECTION & REMOVAL (IQR METHOD)
# ============================================================================

def detect_outliers(df, column='Salary', visualize=True):
    """
    Detect and remove outliers using IQR (Interquartile Range) method.
    
    IQR Method:
    - Q1 (25th percentile)
    - Q3 (75th percentile)
    - IQR = Q3 - Q1
    - Lower Bound = Q1 - 1.5 × IQR
    - Upper Bound = Q3 + 1.5 × IQR
    - Values outside bounds are outliers
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column to check for outliers
        visualize (bool): Whether to create boxplot
        
    Returns:
        pd.DataFrame: Dataframe with outliers removed
    """
    print("\n" + "=" * 70)
    print("STEP 4: OUTLIER DETECTION & REMOVAL (IQR METHOD)".center(70))
    print("=" * 70)
    
    if column not in df.columns:
        print(f"⚠ Column '{column}' not found. Skipping outlier detection.")
        return df
    
    df_no_outliers = df.copy()
    
    # Calculate Q1, Q3, and IQR
    Q1 = df_no_outliers[column].quantile(0.25)
    Q3 = df_no_outliers[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"\n--- IQR Statistics for '{column}' ---")
    print(f"Q1 (25th percentile): {Q1:,.2f}")
    print(f"Q3 (75th percentile): {Q3:,.2f}")
    print(f"IQR (Q3 - Q1): {IQR:,.2f}")
    print(f"Lower Bound: {lower_bound:,.2f}")
    print(f"Upper Bound: {upper_bound:,.2f}")
    
    # Detect outliers
    outliers = df_no_outliers[(df_no_outliers[column] < lower_bound) | 
                               (df_no_outliers[column] > upper_bound)]
    
    print(f"\n✓ Outliers detected: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  Outlier values: {sorted(outliers[column].values)}")
    
    # Remove outliers
    initial_rows = len(df_no_outliers)
    df_no_outliers = df_no_outliers[(df_no_outliers[column] >= lower_bound) & 
                                     (df_no_outliers[column] <= upper_bound)]
    removed_rows = initial_rows - len(df_no_outliers)
    
    print(f"\n✓ Rows removed: {removed_rows}")
    print(f"✓ Remaining rows: {len(df_no_outliers)}")
    
    # Visualization
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Before removing outliers
        axes[0].boxplot(df[column].dropna(), vert=True)
        axes[0].set_title(f'Before Removing Outliers\n({column})', fontsize=12, fontweight='bold')
        axes[0].set_ylabel(column)
        axes[0].grid(True, alpha=0.3)
        
        # After removing outliers
        axes[1].boxplot(df_no_outliers[column].dropna(), vert=True)
        axes[1].set_title(f'After Removing Outliers\n({column})', fontsize=12, fontweight='bold')
        axes[1].set_ylabel(column)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outlier_detection.png', dpi=300, bbox_inches='tight')
        print("\n✓ Boxplot saved as 'outlier_detection.png'")
        plt.show()
    
    print("\n" + "=" * 70)
    print("✓ STEP 4 COMPLETED — OUTLIERS REMOVED".center(70))
    print("=" * 70)
    
    return df_no_outliers


# ============================================================================
# STEP 5: GROUPBY OPERATIONS
# ============================================================================

def groupby_analysis(df):
    """
    Perform GroupBy operations for data aggregation.
    
    Aggregations:
    - Average salary by department
    - Max performance score by city
    - Employee count per department
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing aggregation results
    """
    print("\n" + "=" * 70)
    print("STEP 5: GROUPBY OPERATIONS".center(70))
    print("=" * 70)
    
    results = {}
    
    # Find department columns (original or one-hot encoded)
    dept_cols = [col for col in df.columns if 'Department' in col]
    city_cols = [col for col in df.columns if 'City' in col]
    
    # 1. Average salary by department (if not one-hot encoded)
    if 'Department' in df.columns and 'Salary' in df.columns:
        avg_salary_dept = df.groupby('Department')['Salary'].mean().sort_values(ascending=False)
        results['avg_salary_dept'] = avg_salary_dept
        
        print("\n--- Average Salary by Department ---")
        print(avg_salary_dept.to_frame('Average_Salary'))
    
    # 2. Max performance score by city (if not one-hot encoded)
    if 'City' in df.columns and 'Performance_Score' in df.columns:
        max_perf_city = df.groupby('City')['Performance_Score'].max().sort_values(ascending=False)
        results['max_perf_city'] = max_perf_city
        
        print("\n--- Max Performance Score by City ---")
        print(max_perf_city.to_frame('Max_Performance'))
    
    # 3. Employee count per department
    if 'Department' in df.columns:
        emp_count_dept = df.groupby('Department').size().sort_values(ascending=False)
        results['emp_count_dept'] = emp_count_dept
        
        print("\n--- Employee Count per Department ---")
        print(emp_count_dept.to_frame('Employee_Count'))
    
    # 4. Multiple aggregations
    if 'Department' in df.columns and 'Salary' in df.columns and 'Experience_Years' in df.columns:
        multi_agg = df.groupby('Department').agg({
            'Salary': ['mean', 'min', 'max'],
            'Experience_Years': 'mean',
            'Employee_ID': 'count'
        }).round(2)
        
        print("\n--- Multiple Aggregations by Department ---")
        print(multi_agg)
    
    print("\n" + "=" * 70)
    print("✓ STEP 5 COMPLETED — GROUPBY OPERATIONS DONE".center(70))
    print("=" * 70)
    
    return results


# ============================================================================
# STEP 6: PIVOT & MELT
# ============================================================================

def pivot_melt_ops(df):
    """
    Demonstrate pivot and melt operations.
    
    - Pivot: Reshape data (wide format)
    - Melt: Unpivot data (long format)
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (pivot_table, melted_data)
    """
    print("\n" + "=" * 70)
    print("STEP 6: PIVOT & MELT OPERATIONS".center(70))
    print("=" * 70)
    
    pivot_result = None
    melt_result = None
    
    # Create Pivot Table
    if all(col in df.columns for col in ['Department', 'Gender', 'Salary']):
        print("\n--- Creating Pivot Table ---")
        print("Rows: Department | Columns: Gender | Values: Average Salary")
        
        pivot_result = pd.pivot_table(
            df,
            values='Salary',
            index='Department',
            columns='Gender',
            aggfunc='mean',
            fill_value=0
        ).round(2)
        
        print("\n--- Pivot Table Result ---")
        print(pivot_result)
    
    # Demonstrate Melt (Unpivot)
    if pivot_result is not None:
        print("\n--- Melting Pivot Table (Unpivot) ---")
        
        melt_result = pivot_result.reset_index().melt(
            id_vars='Department',
            var_name='Gender',
            value_name='Average_Salary'
        )
        
        print("\n--- Melted Data (Long Format) ---")
        print(melt_result)
    
    print("\n" + "=" * 70)
    print("✓ STEP 6 COMPLETED — PIVOT & MELT DONE".center(70))
    print("=" * 70)
    
    return pivot_result, melt_result


# ============================================================================
# STEP 7: REMOVE DUPLICATES
# ============================================================================

def remove_duplicates(df, subset=['Employee_ID']):
    """
    Detect and remove duplicate rows.
    
    Args:
        df (pd.DataFrame): Input dataframe
        subset (list): Columns to check for duplicates
        
    Returns:
        pd.DataFrame: Dataframe without duplicates
    """
    print("\n" + "=" * 70)
    print("STEP 7: REMOVE DUPLICATES".center(70))
    print("=" * 70)
    
    df_unique = df.copy()
    
    # Check which columns from subset exist
    valid_subset = [col for col in subset if col in df_unique.columns]
    
    if not valid_subset:
        print("⚠ No valid columns found for duplicate checking.")
        return df_unique
    
    initial_rows = len(df_unique)
    
    # Detect duplicates
    duplicates = df_unique.duplicated(subset=valid_subset, keep=False)
    duplicate_count = duplicates.sum()
    
    print(f"\n--- Duplicate Detection (based on {valid_subset}) ---")
    print(f"Total rows: {initial_rows}")
    print(f"Duplicate rows found: {duplicate_count}")
    
    if duplicate_count > 0:
        print("\n--- Sample Duplicate Rows ---")
        print(df_unique[duplicates].head(10))
    
    # Remove duplicates (keep first occurrence)
    df_unique = df_unique.drop_duplicates(subset=valid_subset, keep='first')
    
    final_rows = len(df_unique)
    removed_rows = initial_rows - final_rows
    
    print(f"\n✓ Duplicates removed: {removed_rows}")
    print(f"✓ Remaining rows: {final_rows}")
    
    print("\n" + "=" * 70)
    print("✓ STEP 7 COMPLETED — DUPLICATES REMOVED".center(70))
    print("=" * 70)
    
    return df_unique


# ============================================================================
# STEP 8: TRAIN-TEST SPLIT
# ============================================================================

def train_test_split_step(df, target_column='Performance_Score', test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Target variable column
        test_size (float): Proportion of test set (0 to 1)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "=" * 70)
    print("STEP 8: TRAIN-TEST SPLIT".center(70))
    print("=" * 70)
    
    if target_column not in df.columns:
        print(f"⚠ Target column '{target_column}' not found.")
        return None, None, None, None
    
    # Prepare features (X) and target (y)
    # Exclude non-numeric columns and target
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
    y = df[target_column]
    
    print(f"\n--- Dataset Split Configuration ---")
    print(f"Target Variable: {target_column}")
    print(f"Features (X): {X.shape[1]} columns")
    print(f"Total Samples: {len(df)}")
    print(f"Train-Test Ratio: {int((1-test_size)*100)}%-{int(test_size*100)}%")
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\n--- Split Results ---")
    print(f"X_train shape: {X_train.shape} — Training features")
    print(f"X_test shape:  {X_test.shape} — Testing features")
    print(f"y_train shape: {y_train.shape} — Training target")
    print(f"y_test shape:  {y_test.shape} — Testing target")
    
    print(f"\n--- Feature Columns ({len(X.columns)}) ---")
    print(list(X.columns))
    
    print("\n--- Training Set Preview ---")
    print(X_train.head(3))
    
    print("\n" + "=" * 70)
    print("✓ STEP 8 COMPLETED — TRAIN-TEST SPLIT DONE".center(70))
    print("=" * 70)
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# BONUS: CORRELATION HEATMAP
# ============================================================================

def create_correlation_heatmap(df):
    """
    Create and save correlation heatmap for numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n" + "=" * 70)
    print("BONUS: CORRELATION HEATMAP".center(70))
    print("=" * 70)
    
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    if numerical_df.shape[1] < 2:
        print("⚠ Not enough numerical columns for correlation analysis.")
        return
    
    # Calculate correlation matrix
    correlation_matrix = numerical_df.corr()
    
    print("\n--- Correlation Matrix ---")
    print(correlation_matrix.round(2))
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'shrink': 0.8}
    )
    plt.title('Correlation Heatmap - Numerical Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n✓ Correlation heatmap saved as 'correlation_heatmap.png'")
    plt.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main function to orchestrate the complete data wrangling pipeline.
    """
    print("\n" + "=" * 70)
    print("DATA WRANGLING TOOLKIT - CORE 8 STEPS".center(70))
    print("=" * 70)
    print("College Lab Experiment - Data Preprocessing".center(70))
    print("=" * 70)
    
    try:
        # Step 0: Load Data
        df = load_data(CSV_FILE)
        
        # Step 1: Handle Missing Values
        df = handle_missing(df)
        
        # Step 7: Remove Duplicates (doing early to clean data)
        df = remove_duplicates(df)
        
        # Step 4: Outlier Detection (before encoding)
        df = detect_outliers(df, column='Salary', visualize=True)
        
        # Store original for pivot/melt operations
        df_original = df.copy()
        
        # Step 5: GroupBy Operations (before encoding)
        groupby_results = groupby_analysis(df_original)
        
        # Step 6: Pivot & Melt (before encoding)
        pivot_melt_ops(df_original)
        
        # Step 2: Encoding Categorical Data
        df = encode_data(df)
        
        # Step 3: Feature Scaling
        df = scale_features(df)
        
        # Bonus: Correlation Heatmap
        create_correlation_heatmap(df)
        
        # Step 8: Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split_step(df)
        
        # Save cleaned dataset
        df.to_csv(OUTPUT_FILE, index=False)
        print("\n" + "=" * 70)
        print(f"✓ Cleaned dataset saved as '{OUTPUT_FILE}'")
        print("=" * 70)
        
        # Final Summary
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY! ✓".center(70))
        print("=" * 70)
        print("\n--- Summary ---")
        print(f"✓ All 8 steps completed successfully")
        print(f"✓ Final dataset shape: {df.shape}")
        print(f"✓ Output file: {OUTPUT_FILE}")
        print(f"✓ Visualizations: outlier_detection.png, correlation_heatmap.png")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR: PIPELINE FAILED ✗".center(70))
        print("=" * 70)
        print(f"Error: {str(e)}")
        print("=" * 70 + "\n")
        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
