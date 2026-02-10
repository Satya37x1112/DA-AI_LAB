# Data Enrichment Using Merge/Join Operations

## 📋 Experiment Overview

**Title:** Data Enrichment Using Merge/Join  
**Aim:** Enrich a primary dataset using a lookup dataset by performing merge/join operations and creating new derived features  
**Tech Stack:** Python, Pandas, NumPy, Matplotlib, Seaborn

---

## 🎯 Learning Objectives

1. Understand different join types (INNER, LEFT, RIGHT)
2. Perform data enrichment using lookup tables
3. Analyze unmatched keys
4. Create derived features
5. Validate merged datasets
6. Export enriched data

---

## 🔧 Installation

### Required Python Packages

```bash
pip install pandas numpy matplotlib seaborn
```

---

## 📁 Project Files

```
AI/
├── experiment8_data_enrichment.py       # Main program
├── employees.csv                        # Primary dataset (53 employees)
├── departments_lookup.csv               # Lookup dataset (7 departments)
├── employees_enriched.csv               # Output (created after running)
├── unmatched_employees.csv              # Unmatched records (if any)
├── data_enrichment_visualizations.png   # Charts
└── DATA_ENRICHMENT_README.md            # This file
```

---

## 🚀 How to Run

1. **Ensure files are in the same directory:**
   - `experiment8_data_enrichment.py`
   - `employees.csv`
   - `departments_lookup.csv`

2. **Run the program:**
   ```bash
   python experiment8_data_enrichment.py
   ```

3. **Check outputs:**
   - Console showing all 8 steps
   - `employees_enriched.csv` - Final enriched dataset
   - `unmatched_employees.csv` - Employees without matching departments
   - `data_enrichment_visualizations.png` - Data insights charts

---

## 📊 Dataset Structure

### Primary Dataset: employees.csv
```csv
Employee_ID,Name,Department_ID,Salary,Experience_Years,City
E001,Rajesh Kumar,D01,75000,5,Mumbai
E002,Priya Sharma,D02,65000,7,Delhi
...
```

**Features:**
- 53 employees
- 6 columns
- Some employees have invalid Department_IDs (D99, D88) for testing unmatched analysis

### Lookup Dataset: departments_lookup.csv
```csv
Department_ID,Department_Name,Department_Budget,Manager_Name
D01,Information Technology,5000000,Rajiv Malhotra
D02,Human Resources,1500000,Sunita Deshmukh
...
```

**Features:**
- 7 departments
- 4 columns
- Complete department information

---

## 🔄 Processing Steps

### 🔹 STEP 1: Load Datasets
- Read both CSV files
- Validate key column existence (Department_ID)
- Display first 5 rows
- Show dataset shapes

### 🔹 STEP 2: Data Cleaning
**Employees Dataset:**
- Remove duplicates
- Drop rows with missing Department_ID
- Convert Department_ID to string and strip spaces
- Strip Name column

**Departments Dataset:**
- Remove duplicate Department_IDs
- Ensure datatype consistency
- Strip all text columns

### 🔹 STEP 3: Merge/Join Operations

#### 1. INNER JOIN
**Definition:** Returns only matched records  
**SQL Equivalent:** `INNER JOIN`  
**Use Case:** Only employees with valid departments

```python
pd.merge(employees, departments, on='Department_ID', how='inner')
```

#### 2. LEFT JOIN
**Definition:** All employees + matched departments  
**SQL Equivalent:** `LEFT JOIN`  
**Use Case:** Keep all employees, even without departments

```python
pd.merge(employees, departments, on='Department_ID', how='left')
```

#### 3. RIGHT JOIN
**Definition:** All departments + matched employees  
**SQL Equivalent:** `RIGHT JOIN`  
**Use Case:** See all departments, including empty ones

```python
pd.merge(employees, departments, on='Department_ID', how='right')
```

**Join Comparison:**
| Join Type | Row Count | Description |
|-----------|-----------|-------------|
| INNER | 50 | Only matched records |
| LEFT | 53 | All employees preserved |
| RIGHT | Variable | All departments preserved |

### 🔹 STEP 4: Unmatched Key Analysis

**Analysis performed:**
- Identify employees without matching departments
- Calculate unmatched percentage
- List unmatched Department_IDs
- Compare with available departments

**Formula:**
```
Unmatched % = (Unmatched Rows / Total Rows) × 100
```

**Example Output:**
```
Total Employees: 53
Matched Employees: 50
Unmatched Employees: 3
Unmatched Percentage: 5.66%
Unmatched Department_IDs: ['D88', 'D99']
```

### 🔹 STEP 5: Data Enrichment

**Columns Added:**
- `Department_Name` - Full department name
- `Department_Budget` - Total budget allocation
- `Manager_Name` - Department manager

**Before:** 6 columns  
**After:** 9 columns

### 🔹 STEP 6: Feature Engineering

#### Feature 1: Salary_to_Budget_Ratio
**Formula:** `(Salary / Department_Budget) × 100`  
**Meaning:** Employee salary as percentage of department budget  
**Range:** 0.0148% to 6.2667%

#### Feature 2: Experience_Level
**Categories:**
- **Junior:** 0-2 years
- **Mid:** 3-5 years
- **Senior:** 6+ years

**Distribution Example:**
```
Mid:     ~20 employees
Senior:  ~18 employees
Junior:  ~12 employees
```

#### Feature 3: Employee_Count
**Per department count** for budget analysis

#### Feature 4: Budget_Per_Employee
**Formula:** `Department_Budget / Employee_Count`  
**Meaning:** Average budget allocation per employee in department  
**Use:** Understand resource distribution

**Final Columns:** 13 total

### 🔹 STEP 7: Data Validation

**Checks performed:**
1. ✅ Null value detection
2. ✅ Row count verification
3. ✅ Employee_ID duplication check
4. ✅ Negative salary check
5. ✅ Negative experience check

### 🔹 STEP 8: Save Output

**Files created:**
1. `employees_enriched.csv` - Complete enriched dataset
2. `unmatched_employees.csv` - Employees without departments (if any)

---

## 📊 Visualizations

### 1. Employee Count by Department
**Type:** Bar chart  
**Shows:** Distribution of employees across departments  
**Insight:** Which departments have most employees

### 2. Salary Distribution by Experience Level
**Type:** Box plot  
**Shows:** Salary ranges for Junior/Mid/Senior levels  
**Insight:** Salary trends with experience

**Saved as:** `data_enrichment_visualizations.png`

---

## 🎓 Key Concepts Explained

### Join Types Comparison

```
Employees (3 records):        Departments (2 records):
E001 → D01                   D01 → IT
E002 → D02                   D02 → HR
E003 → D99 (invalid)

INNER JOIN Result: 2 rows (only E001, E002 matched)
LEFT JOIN Result: 3 rows (all employees kept, E003 has null department)
RIGHT JOIN Result: 2+ rows (all departments kept, may have null employees)
```

### Why Enrichment Matters

**Before Enrichment:**
```
Employee_ID | Department_ID | Salary
E001        | D01           | 75000
```

**After Enrichment:**
```
Employee_ID | Department_ID | Salary | Department_Name | Department_Budget | Manager_Name
E001        | D01           | 75000  | IT              | 5000000          | Rajiv Malhotra
```

**Benefit:** More context for analysis and decision-making!

---

## 💡 Real-World Applications

1. **HR Analytics**
   - Employee-Department analysis
   - Salary benchmarking
   - Resource allocation

2. **Sales Analysis**
   - Customer-Product enrichment
   - Revenue attribution
   - Regional performance

3. **Inventory Management**
   - Product-Supplier linking
   - Stock level analysis
   - Cost optimization

4. **Financial Reporting**
   - Transaction-Account mapping
   - Budget vs. Actual
   - Department-wise expenses

---

## 🔍 Sample Output

```
======================================================================
          DATA ENRICHMENT USING MERGE/JOIN OPERATIONS
======================================================================

--- Join Comparison Summary ---
Join Type  Row Count  Description
INNER      50         Only matched records
LEFT       53         All employees + matched depts
RIGHT      57         All departments + matched emps

--- Unmatched Key Statistics ---
Total Employees: 53
Matched Employees: 50
Unmatched Employees: 3
Unmatched Percentage: 5.66%

--- Feature Engineering Summary ---
Original columns: 9
Final columns: 13
New features added: 4
  • Salary_to_Budget_Ratio
  • Experience_Level
  • Employee_Count
  • Budget_Per_Employee

======================================================================
                PIPELINE COMPLETED SUCCESSFULLY! ✓
======================================================================
```

---

## 🐛 Troubleshooting

**Error: File not found**
```
Ensure employees.csv and departments_lookup.csv are in same directory
```

**Error: Module not found**
```bash
pip install pandas numpy matplotlib seaborn
```

**Error: Key column missing**
```
Ensure both CSVs have 'Department_ID' column
```

**Warning: Unmatched keys**
```
This is expected! Some employees have invalid Department_IDs (D88, D99)
for demonstration purposes
```

---

## 📚 Function Reference

| Function | Purpose |
|----------|---------|
| `load_datasets()` | Load and validate CSV files |
| `clean_data()` | Remove duplicates, handle nulls |
| `perform_joins()` | Execute INNER, LEFT, RIGHT joins |
| `analyze_unmatched_keys()` | Find and analyze unmatched records |
| `enrich_data()` | Add lookup table columns |
| `feature_engineering()` | Create derived features |
| `validate_data()` | Check data quality |
| `save_output()` | Export enriched CSV |
| `create_visualizations()` | Generate charts |

---

## ✨ Extension Ideas

Try enhancing the project:
- Add OUTER JOIN (FULL JOIN)
- Implement fuzzy matching for Department_IDs
- Add more derived features (salary percentiles, tenure categories)
- Create interactive dashboards
- Add data profiling report
- Implement data lineage tracking

---

## 📖 Additional Resources

- [Pandas Merge Documentation](https://pandas.pydata.org/docs/reference/api/pandas.merge.html)
- [SQL Joins Explained](https://www.w3schools.com/sql/sql_join.asp)
- [Feature Engineering Guide](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)

---

## 👨‍💻 Author

Data Engineering Lab Experiment  
Date: February 10, 2026

---

## 📄 License

Educational project for college lab experiments.

---

**Happy Data Engineering! 🔗✨**
