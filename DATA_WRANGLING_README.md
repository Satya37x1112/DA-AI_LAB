# Data Wrangling Toolkit - Core 8 Steps

## 📋 Experiment Overview

**Title:** Data Wrangling Toolkit (Core 8 Steps)  
**Aim:** Practice and implement common data wrangling operations used in real-world data preprocessing  
**Tech Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

## 🎯 Learning Objectives

Master 8 essential data preprocessing techniques:
1. ✅ Handle Missing Values
2. ✅ Encode Categorical Data
3. ✅ Feature Scaling
4. ✅ Outlier Detection & Removal
5. ✅ GroupBy Operations
6. ✅ Pivot & Melt
7. ✅ Remove Duplicates
8. ✅ Train-Test Split

---

## 🔧 Installation

### Required Python Packages

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Package Purposes:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities (encoding, scaling, splitting)
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

---

## 📁 Project Files

```
AI/
├── experiment7_data_wrangling_toolkit.py    # Main program
├── employees.csv                            # Sample dataset
├── employees_cleaned.csv                    # Output (created after running)
├── outlier_detection.png                    # Boxplot visualization
├── correlation_heatmap.png                  # Correlation matrix
└── DATA_WRANGLING_README.md                 # This file
```

---

## 🚀 How to Run

1. **Ensure files are in the same directory:**
   - `experiment7_data_wrangling_toolkit.py`
   - `employees.csv`

2. **Run the program:**
   ```bash
   python experiment7_data_wrangling_toolkit.py
   ```

3. **Check outputs:**
   - Console output showing each processing step
   - `employees_cleaned.csv` - Processed dataset
   - `outlier_detection.png` - Before/after boxplot
   - `correlation_heatmap.png` - Feature correlation matrix

---

## 📊 Sample Dataset Structure

### employees.csv
```csv
Employee_ID,Name,Age,Gender,Department,Salary,Experience_Years,City,Performance_Score
E001,Rajesh Kumar,28,Male,IT,75000,5,Mumbai,8.5
E002,Priya Sharma,32,Female,HR,65000,7,Delhi,7.8
...
```

### Dataset Features:
- **50 employees** with intentional data quality issues
- **Missing values** in Age, Gender, Experience_Years, Salary, Name
- **Outliers** in Salary column (extremely high/low values)
- **Duplicate records** (Employee_ID E001 and E019)
- **9 columns** including numerical and categorical data

---

## 🔄 Processing Steps Explained

### 🔹 STEP 1: Handle Missing Values
**Purpose:** Clean incomplete data

**Techniques:**
- Detect nulls column-wise
- Fill numerical columns with **median**
- Fill categorical columns with **mode**
- Drop rows with >50% missing values

**Example:**
```python
# Before: Age has 2 missing values
# After: Filled with median age (31)
```

---

### 🔹 STEP 2: Encoding Categorical Data
**Purpose:** Convert text to numbers for ML models

**Why needed?** Machine learning algorithms only understand numbers!

**Techniques:**
- **Label Encoding** for Gender (Male→0, Female→1)
- **One-Hot Encoding** for Department & City (creates binary columns)

**Example:**
```python
# Before: Gender = ["Male", "Female"]
# After: Gender_Encoded = [0, 1]

# Before: Department = "IT"
# After: Department_IT = 1, Department_HR = 0, ...
```

---

### 🔹 STEP 3: Feature Scaling
**Purpose:** Normalize data ranges for better model performance

**Techniques:**
1. **Min-Max Scaling** → Scales to [0, 1] range
2. **Standardization** → Scales to mean=0, std=1

**Example:**
```python
# Original Salary: 15,000 to 450,000
# Min-Max: 0.0 to 1.0
# Standard: -1.8 to 2.5
```

---

### 🔹 STEP 4: Outlier Detection & Removal (IQR)
**Purpose:** Remove extreme values that can skew analysis

**IQR Method:**
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

**Visualization:** Boxplot showing before/after outlier removal

---

### 🔹 STEP 5: GroupBy Operations
**Purpose:** Aggregate data for insights

**Analyses performed:**
- Average salary by department
- Max performance score by city
- Employee count per department
- Multiple aggregations (mean, min, max)

**Example Output:**
```
Department    Average_Salary
IT            85,234.56
Finance       92,150.00
Sales         84,500.00
```

---

### 🔹 STEP 6: Pivot & Melt
**Purpose:** Reshape data for different analysis needs

**Pivot Table:**
- Rows: Department
- Columns: Gender
- Values: Average Salary

**Melt:** Converts wide format → long format (unpivot)

---

### 🔹 STEP 7: Remove Duplicates
**Purpose:** Ensure data uniqueness

**Process:**
- Detect duplicates based on Employee_ID
- Keep first occurrence
- Remove subsequent duplicates

**Example:**
```
Before: 50 rows (with duplicates)
After: 49 rows (duplicates removed)
```

---

### 🔹 STEP 8: Train-Test Split
**Purpose:** Prepare data for machine learning

**Configuration:**
- Features (X): All numerical columns except target
- Target (y): Performance_Score
- Split ratio: 80% train, 20% test
- Random state: 42 (for reproducibility)

**Output:**
```
X_train: (40, 15) — 40 samples, 15 features
X_test:  (10, 15) — 10 samples, 15 features
y_train: (40,)
y_test:  (10,)
```

---

## 📈 Expected Output Structure

```
======================================================================
                  DATA WRANGLING TOOLKIT - CORE 8 STEPS
======================================================================

======================================================================
                         STEP 0: LOADING DATA
======================================================================
✓ Data loaded successfully from 'employees.csv'
✓ Shape: 50 rows × 9 columns
...

======================================================================
              ✓ STEP 1 COMPLETED — MISSING VALUES HANDLED
======================================================================

[Continues for all 8 steps...]

======================================================================
                  PIPELINE COMPLETED SUCCESSFULLY! ✓
======================================================================
```

---

## 📊 Visualizations Generated

### 1. Outlier Detection Boxplot
- **File:** `outlier_detection.png`
- **Content:** Side-by-side boxplots showing salary distribution before and after outlier removal

### 2. Correlation Heatmap
- **File:** `correlation_heatmap.png`
- **Content:** Color-coded matrix showing relationships between numerical features
- **Colors:** Red (positive correlation), Blue (negative correlation)

---

## 🎓 Key Concepts Learned

### Data Quality Issues:
- Missing values
- Duplicates
- Outliers
- Inconsistent formatting

### Data Transformation:
- Encoding (Label & One-Hot)
- Scaling (Min-Max & Standard)
- Aggregation (GroupBy)
- Reshaping (Pivot & Melt)

### ML Preparation:
- Feature selection
- Train-test splitting
- Data validation

---

## 🐛 Troubleshooting

**Error: File not found**
```
Ensure 'employees.csv' is in the same directory as the Python script
```

**Error: Module not found**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Error: No display for plots**
```python
# Plots are saved as PNG files even if display fails
# Check: outlier_detection.png, correlation_heatmap.png
```

**Warning: SettingWithCopyWarning**
```
This is handled in the code using .copy() to avoid warnings
```

---

## 📚 Function Reference

| Function | Purpose |
|----------|---------|
| `load_data()` | Load and validate CSV data |
| `handle_missing()` | Fill/remove missing values |
| `encode_data()` | Convert categorical to numerical |
| `scale_features()` | Normalize feature ranges |
| `detect_outliers()` | Find and remove outliers using IQR |
| `groupby_analysis()` | Aggregate data by groups |
| `pivot_melt_ops()` | Reshape data (wide ↔ long) |
| `remove_duplicates()` | Remove duplicate records |
| `train_test_split_step()` | Split data for ML |
| `create_correlation_heatmap()` | Visualize feature relationships |

---

## 🔍 Data Quality Checks

The program validates:
- ✅ File existence
- ✅ Non-empty dataset
- ✅ Column existence before operations
- ✅ Data type compatibility
- ✅ Sufficient data for operations

---

## 💡 Best Practices Demonstrated

1. **Modular Code:** Each step is a separate function
2. **Error Handling:** Try-except blocks for robustness
3. **Documentation:** Comprehensive docstrings
4. **Validation:** Check data before operations
5. **Visualization:** Visual inspection of transformations
6. **Reproducibility:** Fixed random seeds
7. **Output Files:** Save processed data and plots

---

## 📖 Additional Learning Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Data Wrangling with Python](https://realpython.com/pandas-python-explore-dataset/)
- [Feature Engineering Guide](https://towardsdatascience.com/tagged/feature-engineering)

---

## 🎯 Practical Applications

These techniques are used in:
- 📊 Data Analysis Projects
- 🤖 Machine Learning Pipelines
- 📈 Business Intelligence
- 🔬 Research Data Processing
- 💼 Industry Analytics

---

## ✨ Extension Ideas

Try enhancing the project:
- Add more encoding methods (Target Encoding, Frequency Encoding)
- Implement SMOTE for imbalanced data
- Add feature importance analysis
- Create interactive visualizations with Plotly
- Build a simple ML model using the processed data

---

## 👨‍💻 Author

Data Analytics Lab Experiment  
Date: February 10, 2026

---

## 📄 License

Educational project for college lab experiments.

---

**Happy Data Wrangling! 📊✨**
