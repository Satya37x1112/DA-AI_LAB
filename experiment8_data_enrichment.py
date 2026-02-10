"""
Data Enrichment Using Merge/Join Operations
============================================
Experiment Title: Data Enrichment Using Merge/Join

Aim: Enrich a primary dataset using a lookup dataset by performing 
     merge/join operations and creating new derived features.

Author: Data Engineering Lab
Date: February 10, 2026

This program demonstrates:
- Multiple join types (INNER, LEFT, RIGHT)
- Unmatched key analysis
- Data enrichment
- Feature engineering
- Data validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# CONFIGURATION
# ============================================================================

EMPLOYEES_CSV = 'employees.csv'
DEPARTMENTS_CSV = 'departments_lookup.csv'
OUTPUT_FILE = 'employees_enriched.csv'
UNMATCHED_FILE = 'unmatched_employees.csv'


# ============================================================================
# STEP 1: LOAD DATASETS
# ============================================================================

def load_datasets():
    """
    Load primary and lookup datasets from CSV files.
    
    Returns:
        tuple: (employees_df, departments_df)
        
    Raises:
        FileNotFoundError: If CSV files don't exist
        ValueError: If required columns are missing
    """
    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATASETS".center(70))
    print("=" * 70)
    
    try:
        # Load employees dataset (primary)
        if not os.path.exists(EMPLOYEES_CSV):
            raise FileNotFoundError(f"Primary dataset '{EMPLOYEES_CSV}' not found!")
        
        employees_df = pd.read_csv(EMPLOYEES_CSV)
        print(f"\n✓ Loaded primary dataset: {EMPLOYEES_CSV}")
        print(f"  Shape: {employees_df.shape[0]} rows × {employees_df.shape[1]} columns")
        
        # Load departments lookup dataset
        if not os.path.exists(DEPARTMENTS_CSV):
            raise FileNotFoundError(f"Lookup dataset '{DEPARTMENTS_CSV}' not found!")
        
        departments_df = pd.read_csv(DEPARTMENTS_CSV)
        print(f"\n✓ Loaded lookup dataset: {DEPARTMENTS_CSV}")
        print(f"  Shape: {departments_df.shape[0]} rows × {departments_df.shape[1]} columns")
        
        # Validate key column existence
        if 'Department_ID' not in employees_df.columns:
            raise ValueError("'Department_ID' column missing in employees dataset!")
        
        if 'Department_ID' not in departments_df.columns:
            raise ValueError("'Department_ID' column missing in departments dataset!")
        
        print(f"\n✓ Key column 'Department_ID' validated in both datasets")
        
        # Display first 5 rows of each dataset
        print("\n--- Employees Dataset (First 5 Rows) ---")
        print(employees_df.head())
        
        print("\n--- Departments Lookup Dataset (First 5 Rows) ---")
        print(departments_df.head())
        
        # Display dataset info
        print("\n--- Employees Dataset Info ---")
        print(f"Columns: {list(employees_df.columns)}")
        print(f"Dtypes: {dict(employees_df.dtypes)}")
        
        print("\n--- Departments Dataset Info ---")
        print(f"Columns: {list(departments_df.columns)}")
        print(f"Dtypes: {dict(departments_df.dtypes)}")
        
        print("\n" + "=" * 70)
        print("✓ DATASETS LOADED SUCCESSFULLY".center(70))
        print("=" * 70)
        
        return employees_df, departments_df
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        raise
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


# ============================================================================
# STEP 2: DATA CLEANING
# ============================================================================

def clean_data(employees_df, departments_df):
    """
    Clean both datasets before merging.
    
    - Remove duplicates
    - Handle missing Department_ID values
    - Ensure datatype consistency
    - Strip spaces in key fields
    
    Args:
        employees_df (pd.DataFrame): Employees dataset
        departments_df (pd.DataFrame): Departments dataset
        
    Returns:
        tuple: (cleaned_employees_df, cleaned_departments_df)
    """
    print("\n" + "=" * 70)
    print("STEP 2: DATA CLEANING".center(70))
    print("=" * 70)
    
    emp_clean = employees_df.copy()
    dept_clean = departments_df.copy()
    
    # Clean Employees Dataset
    print("\n--- Cleaning Employees Dataset ---")
    
    # 1. Remove duplicates
    emp_initial = len(emp_clean)
    emp_clean = emp_clean.drop_duplicates()
    emp_removed = emp_initial - len(emp_clean)
    print(f"✓ Duplicates removed: {emp_removed}")
    
    # 2. Handle missing Department_ID
    missing_dept_id = emp_clean['Department_ID'].isnull().sum()
    print(f"✓ Missing Department_ID values: {missing_dept_id}")
    
    if missing_dept_id > 0:
        print(f"  → Dropping {missing_dept_id} rows with missing Department_ID")
        emp_clean = emp_clean.dropna(subset=['Department_ID'])
    
    # 3. Ensure datatype consistency and strip spaces
    emp_clean['Department_ID'] = emp_clean['Department_ID'].astype(str).str.strip()
    print(f"✓ Department_ID converted to string and stripped")
    
    # 4. Strip spaces from Name column if exists
    if 'Name' in emp_clean.columns:
        emp_clean['Name'] = emp_clean['Name'].str.strip()
        print(f"✓ Name column stripped of extra spaces")
    
    # Clean Departments Dataset
    print("\n--- Cleaning Departments Lookup Dataset ---")
    
    # 1. Remove duplicates
    dept_initial = len(dept_clean)
    dept_clean = dept_clean.drop_duplicates(subset=['Department_ID'])
    dept_removed = dept_initial - len(dept_clean)
    print(f"✓ Duplicates removed: {dept_removed}")
    
    # 2. Handle missing Department_ID
    missing_dept_id_lookup = dept_clean['Department_ID'].isnull().sum()
    print(f"✓ Missing Department_ID values: {missing_dept_id_lookup}")
    
    if missing_dept_id_lookup > 0:
        dept_clean = dept_clean.dropna(subset=['Department_ID'])
    
    # 3. Ensure datatype consistency and strip spaces
    dept_clean['Department_ID'] = dept_clean['Department_ID'].astype(str).str.strip()
    print(f"✓ Department_ID converted to string and stripped")
    
    # 4. Strip spaces from text columns
    for col in dept_clean.select_dtypes(include=['object']).columns:
        dept_clean[col] = dept_clean[col].str.strip()
    print(f"✓ All text columns stripped of extra spaces")
    
    # Summary
    print(f"\n--- Cleaning Summary ---")
    print(f"Employees: {emp_initial} → {len(emp_clean)} rows")
    print(f"Departments: {dept_initial} → {len(dept_clean)} rows")
    
    print("\n" + "=" * 70)
    print("✓ DATA CLEANING COMPLETED".center(70))
    print("=" * 70)
    
    return emp_clean, dept_clean


# ============================================================================
# STEP 3: MERGE / JOIN OPERATIONS
# ============================================================================

def perform_joins(employees_df, departments_df):
    """
    Perform multiple join operations to demonstrate differences.
    
    Join Types:
    1. INNER JOIN → Only matched records (intersection)
    2. LEFT JOIN → All employees, matched departments
    3. RIGHT JOIN → All departments, matched employees
    
    Args:
        employees_df (pd.DataFrame): Cleaned employees dataset
        departments_df (pd.DataFrame): Cleaned departments dataset
        
    Returns:
        dict: Dictionary containing all join results
    """
    print("\n" + "=" * 70)
    print("STEP 3: MERGE / JOIN OPERATIONS".center(70))
    print("=" * 70)
    
    join_results = {}
    
    # -------------------------------------------------------------------------
    # 1. INNER JOIN
    # -------------------------------------------------------------------------
    print("\n--- 1. INNER JOIN ---")
    print("Definition: Returns only rows where Department_ID exists in BOTH datasets")
    print("Use case: When you only want employees with valid departments")
    
    inner_join = pd.merge(
        employees_df,
        departments_df,
        on='Department_ID',
        how='inner'
    )
    
    join_results['inner'] = inner_join
    print(f"\n✓ Result: {len(inner_join)} rows")
    print(f"  (Only employees with matching departments)")
    
    # -------------------------------------------------------------------------
    # 2. LEFT JOIN
    # -------------------------------------------------------------------------
    print("\n--- 2. LEFT JOIN ---")
    print("Definition: Returns ALL employees, with department info where available")
    print("Use case: Keep all employees, even without valid departments")
    
    left_join = pd.merge(
        employees_df,
        departments_df,
        on='Department_ID',
        how='left'
    )
    
    join_results['left'] = left_join
    print(f"\n✓ Result: {len(left_join)} rows")
    print(f"  (All employees preserved)")
    
    # Count unmatched in left join
    unmatched_left = left_join['Department_Name'].isnull().sum()
    print(f"  Employees without matching department: {unmatched_left}")
    
    # -------------------------------------------------------------------------
    # 3. RIGHT JOIN
    # -------------------------------------------------------------------------
    print("\n--- 3. RIGHT JOIN ---")
    print("Definition: Returns ALL departments, with employees where available")
    print("Use case: See all departments, even those without employees")
    
    right_join = pd.merge(
        employees_df,
        departments_df,
        on='Department_ID',
        how='right'
    )
    
    join_results['right'] = right_join
    print(f"\n✓ Result: {len(right_join)} rows")
    print(f"  (All departments preserved)")
    
    # Count departments without employees
    unmatched_right = right_join['Name'].isnull().sum()
    print(f"  Departments without employees: {unmatched_right}")
    
    # -------------------------------------------------------------------------
    # Join Comparison Summary
    # -------------------------------------------------------------------------
    print("\n--- Join Comparison Summary ---")
    comparison_df = pd.DataFrame({
        'Join Type': ['INNER', 'LEFT', 'RIGHT'],
        'Row Count': [len(inner_join), len(left_join), len(right_join)],
        'Description': [
            'Only matched records',
            'All employees + matched depts',
            'All departments + matched emps'
        ]
    })
    print(comparison_df.to_string(index=False))
    
    # Display sample of inner join
    print("\n--- Sample of INNER JOIN Result (First 5 Rows) ---")
    print(inner_join.head())
    
    print("\n" + "=" * 70)
    print("✓ JOIN OPERATIONS COMPLETED".center(70))
    print("=" * 70)
    
    return join_results


# ============================================================================
# STEP 4: UNMATCHED KEY ANALYSIS
# ============================================================================

def analyze_unmatched_keys(employees_df, departments_df, left_join_df):
    """
    Analyze unmatched Department_IDs and calculate statistics.
    
    Args:
        employees_df (pd.DataFrame): Employees dataset
        departments_df (pd.DataFrame): Departments dataset
        left_join_df (pd.DataFrame): Result of left join
        
    Returns:
        pd.DataFrame: Unmatched employees dataset
    """
    print("\n" + "=" * 70)
    print("STEP 4: UNMATCHED KEY ANALYSIS".center(70))
    print("=" * 70)
    
    # Find unmatched employees (those without department match)
    unmatched_mask = left_join_df['Department_Name'].isnull()
    unmatched_employees = left_join_df[unmatched_mask].copy()
    
    # Count statistics
    total_employees = len(employees_df)
    unmatched_count = len(unmatched_employees)
    matched_count = total_employees - unmatched_count
    unmatched_percentage = (unmatched_count / total_employees) * 100 if total_employees > 0 else 0
    
    print("\n--- Unmatched Key Statistics ---")
    print(f"Total Employees: {total_employees}")
    print(f"Matched Employees: {matched_count}")
    print(f"Unmatched Employees: {unmatched_count}")
    print(f"Unmatched Percentage: {unmatched_percentage:.2f}%")
    
    # Show unmatched Department_IDs
    if unmatched_count > 0:
        unmatched_dept_ids = unmatched_employees['Department_ID'].unique()
        print(f"\n--- Unmatched Department_IDs ---")
        print(f"Unique unmatched IDs: {list(unmatched_dept_ids)}")
        
        print(f"\n--- Sample Unmatched Employees ---")
        print(unmatched_employees[['Employee_ID', 'Name', 'Department_ID', 'Salary']].head(10))
    else:
        print("\n✓ All employees have matching departments!")
    
    # Compare with available departments
    available_dept_ids = departments_df['Department_ID'].unique()
    print(f"\n--- Available Department_IDs in Lookup ---")
    print(f"Count: {len(available_dept_ids)}")
    print(f"IDs: {sorted(available_dept_ids)}")
    
    # Find Department_IDs in employees but not in lookup
    emp_dept_ids = set(employees_df['Department_ID'].unique())
    lookup_dept_ids = set(departments_df['Department_ID'].unique())
    missing_in_lookup = emp_dept_ids - lookup_dept_ids
    
    if missing_in_lookup:
        print(f"\n⚠ Department_IDs in employees but NOT in lookup:")
        print(f"  {sorted(missing_in_lookup)}")
    
    print("\n" + "=" * 70)
    print("✓ UNMATCHED KEY ANALYSIS DONE".center(70))
    print("=" * 70)
    
    return unmatched_employees


# ============================================================================
# STEP 5: DATA ENRICHMENT
# ============================================================================

def enrich_data(employees_df, departments_df):
    """
    Enrich employee data with department information.
    
    Adds columns:
    - Department_Name
    - Department_Budget
    - Manager_Name
    
    Args:
        employees_df (pd.DataFrame): Employees dataset
        departments_df (pd.DataFrame): Departments dataset
        
    Returns:
        pd.DataFrame: Enriched dataset
    """
    print("\n" + "=" * 70)
    print("STEP 5: DATA ENRICHMENT".center(70))
    print("=" * 70)
    
    print("\n--- Original Employees Columns ---")
    print(list(employees_df.columns))
    
    print("\n--- Enrichment Source (Departments Lookup) ---")
    print(list(departments_df.columns))
    
    # Perform left join to enrich
    enriched_df = pd.merge(
        employees_df,
        departments_df,
        on='Department_ID',
        how='left'
    )
    
    print("\n--- New Columns Added After Enrichment ---")
    new_columns = set(enriched_df.columns) - set(employees_df.columns)
    print(list(new_columns))
    
    print("\n--- Enriched Dataset Columns ---")
    print(list(enriched_df.columns))
    
    print(f"\n✓ Dataset enriched: {len(employees_df.columns)} → {len(enriched_df.columns)} columns")
    
    # Show sample enriched data
    print("\n--- Sample Enriched Data (First 5 Rows) ---")
    print(enriched_df.head())
    
    print("\n" + "=" * 70)
    print("✓ DATA ENRICHMENT COMPLETED".center(70))
    print("=" * 70)
    
    return enriched_df


# ============================================================================
# STEP 6: FEATURE ENGINEERING
# ============================================================================

def feature_engineering(enriched_df):
    """
    Create new derived features from enriched dataset.
    
    New Features:
    1. Salary_to_Budget_Ratio: Employee salary as % of dept budget
    2. Experience_Level: Junior/Mid/Senior based on experience
    3. Budget_Per_Employee: Average budget allocation per employee
    
    Args:
        enriched_df (pd.DataFrame): Enriched dataset
        
    Returns:
        pd.DataFrame: Dataset with engineered features
    """
    print("\n" + "=" * 70)
    print("STEP 6: FEATURE ENGINEERING".center(70))
    print("=" * 70)
    
    df_engineered = enriched_df.copy()
    
    # -------------------------------------------------------------------------
    # Feature 1: Salary_to_Budget_Ratio
    # -------------------------------------------------------------------------
    print("\n--- Feature 1: Salary_to_Budget_Ratio ---")
    print("Formula: Salary / Department_Budget")
    print("Meaning: What % of department budget is this employee's salary")
    
    # Protect against division by zero
    df_engineered['Salary_to_Budget_Ratio'] = np.where(
        df_engineered['Department_Budget'] > 0,
        (df_engineered['Salary'] / df_engineered['Department_Budget']) * 100,
        0
    )
    
    print(f"✓ Created: Salary_to_Budget_Ratio (in %)")
    print(f"  Range: {df_engineered['Salary_to_Budget_Ratio'].min():.4f}% to {df_engineered['Salary_to_Budget_Ratio'].max():.4f}%")
    
    # -------------------------------------------------------------------------
    # Feature 2: Experience_Level
    # -------------------------------------------------------------------------
    print("\n--- Feature 2: Experience_Level ---")
    print("Categories:")
    print("  0-2 years  → Junior")
    print("  3-5 years  → Mid")
    print("  6+ years   → Senior")
    
    def categorize_experience(years):
        """Categorize experience into levels."""
        if pd.isna(years):
            return 'Unknown'
        elif years <= 2:
            return 'Junior'
        elif years <= 5:
            return 'Mid'
        else:
            return 'Senior'
    
    df_engineered['Experience_Level'] = df_engineered['Experience_Years'].apply(categorize_experience)
    
    print(f"✓ Created: Experience_Level")
    print(f"\n  Distribution:")
    print(df_engineered['Experience_Level'].value_counts().to_frame('Count'))
    
    # -------------------------------------------------------------------------
    # Feature 3: Budget_Per_Employee (GroupBy)
    # -------------------------------------------------------------------------
    print("\n--- Feature 3: Budget_Per_Employee ---")
    print("Formula: Department_Budget / Employee_Count_in_Department")
    print("Meaning: Average budget allocated per employee in department")
    
    # Calculate employee count per department
    dept_emp_count = df_engineered.groupby('Department_ID').size().reset_index(name='Employee_Count')
    
    # Merge back
    df_engineered = pd.merge(
        df_engineered,
        dept_emp_count,
        on='Department_ID',
        how='left'
    )
    
    # Calculate budget per employee
    df_engineered['Budget_Per_Employee'] = np.where(
        df_engineered['Employee_Count'] > 0,
        df_engineered['Department_Budget'] / df_engineered['Employee_Count'],
        0
    )
    
    print(f"✓ Created: Budget_Per_Employee")
    print(f"  Range: ${df_engineered['Budget_Per_Employee'].min():,.2f} to ${df_engineered['Budget_Per_Employee'].max():,.2f}")
    
    # Summary
    print("\n--- Feature Engineering Summary ---")
    print(f"Original columns: {len(enriched_df.columns)}")
    print(f"Final columns: {len(df_engineered.columns)}")
    print(f"New features added: 4")
    print(f"  • Salary_to_Budget_Ratio")
    print(f"  • Experience_Level")
    print(f"  • Employee_Count")
    print(f"  • Budget_Per_Employee")
    
    # Show sample
    print("\n--- Sample with Engineered Features ---")
    display_cols = ['Name', 'Salary', 'Department_Budget', 'Salary_to_Budget_Ratio', 
                    'Experience_Years', 'Experience_Level', 'Budget_Per_Employee']
    print(df_engineered[display_cols].head())
    
    print("\n" + "=" * 70)
    print("✓ FEATURE ENGINEERING COMPLETED".center(70))
    print("=" * 70)
    
    return df_engineered


# ============================================================================
# STEP 7: DATA VALIDATION
# ============================================================================

def validate_data(df):
    """
    Validate the final enriched and engineered dataset.
    
    Checks:
    - Null values
    - Row counts
    - Key duplication
    - Data integrity
    
    Args:
        df (pd.DataFrame): Final dataset
        
    Returns:
        bool: True if validation passes
    """
    print("\n" + "=" * 70)
    print("STEP 7: DATA VALIDATION".center(70))
    print("=" * 70)
    
    validation_passed = True
    
    # -------------------------------------------------------------------------
    # 1. Check for null values
    # -------------------------------------------------------------------------
    print("\n--- 1. Null Value Check ---")
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    
    if len(null_counts) > 0:
        print("⚠ Columns with null values:")
        print(null_counts.to_frame('Null_Count'))
    else:
        print("✓ No null values found")
    
    # -------------------------------------------------------------------------
    # 2. Row count verification
    # -------------------------------------------------------------------------
    print("\n--- 2. Row Count Verification ---")
    row_count = len(df)
    print(f"✓ Total rows in final dataset: {row_count}")
    
    if row_count == 0:
        print("✗ Dataset is empty!")
        validation_passed = False
    
    # -------------------------------------------------------------------------
    # 3. Key duplication check
    # -------------------------------------------------------------------------
    print("\n--- 3. Employee_ID Duplication Check ---")
    duplicates = df.duplicated(subset=['Employee_ID'], keep=False).sum()
    
    if duplicates > 0:
        print(f"⚠ Duplicate Employee_IDs found: {duplicates}")
        validation_passed = False
    else:
        print("✓ No duplicate Employee_IDs")
    
    # -------------------------------------------------------------------------
    # 4. Data integrity checks
    # -------------------------------------------------------------------------
    print("\n--- 4. Data Integrity Checks ---")
    
    # Check negative salaries
    if (df['Salary'] < 0).any():
        print("⚠ Negative salaries found")
        validation_passed = False
    else:
        print("✓ All salaries are positive")
    
    # Check negative experience
    if (df['Experience_Years'] < 0).any():
        print("⚠ Negative experience years found")
        validation_passed = False
    else:
        print("✓ All experience values are valid")
    
    # -------------------------------------------------------------------------
    # 5. Dataset summary
    # -------------------------------------------------------------------------
    print("\n--- 5. Final Dataset Summary ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    print("\n--- Column List ---")
    print(list(df.columns))
    
    # Final validation result
    print("\n" + "=" * 70)
    if validation_passed:
        print("✓ DATA VALIDATION PASSED".center(70))
    else:
        print("⚠ DATA VALIDATION FAILED - CHECK WARNINGS ABOVE".center(70))
    print("=" * 70)
    
    return validation_passed


# ============================================================================
# STEP 8: SAVE OUTPUT
# ============================================================================

def save_output(enriched_df, unmatched_df):
    """
    Save enriched dataset and unmatched records to CSV files.
    
    Args:
        enriched_df (pd.DataFrame): Final enriched dataset
        unmatched_df (pd.DataFrame): Unmatched employees
    """
    print("\n" + "=" * 70)
    print("STEP 8: SAVE OUTPUT".center(70))
    print("=" * 70)
    
    # Save enriched dataset
    enriched_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Enriched dataset saved: {OUTPUT_FILE}")
    print(f"  Rows: {len(enriched_df)}")
    print(f"  Columns: {len(enriched_df.columns)}")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")
    
    # Save unmatched records (if any)
    if len(unmatched_df) > 0:
        unmatched_df.to_csv(UNMATCHED_FILE, index=False)
        print(f"\n✓ Unmatched records saved: {UNMATCHED_FILE}")
        print(f"  Rows: {len(unmatched_df)}")
    else:
        print(f"\n✓ No unmatched records to save")
    
    print("\n" + "=" * 70)
    print("✓ OUTPUT SAVED SUCCESSFULLY".center(70))
    print("=" * 70)


# ============================================================================
# BONUS: VISUALIZATIONS
# ============================================================================

def create_visualizations(enriched_df):
    """
    Create visualizations for data insights.
    
    1. Employee count by department (bar chart)
    2. Salary distribution by experience level (box plot)
    
    Args:
        enriched_df (pd.DataFrame): Enriched dataset
    """
    print("\n" + "=" * 70)
    print("BONUS: CREATING VISUALIZATIONS".center(70))
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Visualization 1: Employee Count by Department
    # -------------------------------------------------------------------------
    print("\n--- Creating Visualization 1: Employee Count by Department ---")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count employees per department
    dept_counts = enriched_df['Department_Name'].value_counts().sort_values(ascending=False)
    
    # Bar chart
    axes[0].bar(range(len(dept_counts)), dept_counts.values, color='steelblue', edgecolor='black')
    axes[0].set_xticks(range(len(dept_counts)))
    axes[0].set_xticklabels(dept_counts.index, rotation=45, ha='right')
    axes[0].set_xlabel('Department', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Employee Count', fontsize=12, fontweight='bold')
    axes[0].set_title('Employee Distribution by Department', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(dept_counts.values):
        axes[0].text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Visualization 2: Salary Distribution by Experience Level
    # -------------------------------------------------------------------------
    print("--- Creating Visualization 2: Salary by Experience Level ---")
    
    # Box plot
    experience_order = ['Junior', 'Mid', 'Senior']
    data_to_plot = [enriched_df[enriched_df['Experience_Level'] == level]['Salary'].dropna() 
                    for level in experience_order if level in enriched_df['Experience_Level'].values]
    labels_to_plot = [level for level in experience_order if level in enriched_df['Experience_Level'].values]
    
    bp = axes[1].boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                          boxprops=dict(facecolor='lightcoral', color='black'),
                          medianprops=dict(color='darkred', linewidth=2),
                          whiskerprops=dict(color='black'),
                          capprops=dict(color='black'))
    
    axes[1].set_xlabel('Experience Level', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Salary ($)', fontsize=12, fontweight='bold')
    axes[1].set_title('Salary Distribution by Experience Level', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('data_enrichment_visualizations.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualizations saved: data_enrichment_visualizations.png")
    plt.close()
    
    print("\n" + "=" * 70)
    print("✓ VISUALIZATIONS COMPLETED".center(70))
    print("=" * 70)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main function to orchestrate the complete data enrichment pipeline.
    """
    print("\n" + "=" * 70)
    print("DATA ENRICHMENT USING MERGE/JOIN OPERATIONS".center(70))
    print("=" * 70)
    print("College Lab Experiment - Data Enrichment".center(70))
    print("=" * 70)
    
    try:
        # Step 1: Load datasets
        employees_df, departments_df = load_datasets()
        
        # Step 2: Clean data
        employees_clean, departments_clean = clean_data(employees_df, departments_df)
        
        # Step 3: Perform joins
        join_results = perform_joins(employees_clean, departments_clean)
        
        # Step 4: Analyze unmatched keys
        unmatched_employees = analyze_unmatched_keys(
            employees_clean,
            departments_clean,
            join_results['left']
        )
        
        # Step 5: Enrich data
        enriched_df = enrich_data(employees_clean, departments_clean)
        
        # Step 6: Feature engineering
        final_df = feature_engineering(enriched_df)
        
        # Step 7: Validate data
        validation_passed = validate_data(final_df)
        
        # Step 8: Save output
        save_output(final_df, unmatched_employees)
        
        # Bonus: Visualizations
        create_visualizations(final_df)
        
        # Final summary
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY! ✓".center(70))
        print("=" * 70)
        print("\n--- Final Summary ---")
        print(f"✓ All 8 steps completed successfully")
        print(f"✓ Original employees: {len(employees_df)}")
        print(f"✓ Final enriched dataset: {len(final_df)} rows × {len(final_df.columns)} columns")
        print(f"✓ Unmatched employees: {len(unmatched_employees)}")
        print(f"✓ Output files created:")
        print(f"  • {OUTPUT_FILE}")
        if len(unmatched_employees) > 0:
            print(f"  • {UNMATCHED_FILE}")
        print(f"  • data_enrichment_visualizations.png")
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
