# 🎓 Data Engineering & Analytics Lab Experiments

> **Complete Collection of Python Data Science Experiments**  
> Perfect for college lab assignments, data engineering practice, and hands-on learning

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Experiments List](#experiments-list)
4. [Quick Start Guide](#quick-start-guide)
5. [Dataset Files](#dataset-files)
6. [Output Files](#output-files)
7. [Running the Experiments](#running-the-experiments)
8. [Learning Path](#learning-path)

---

## 🎯 Overview

This repository contains **9 complete Python experiments** covering essential data engineering and data analytics concepts. Each experiment is a standalone, fully-functional program with comprehensive documentation, sample datasets, and detailed outputs.

**Total Lines of Code:** ~8,200+  
**Technologies:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SQLite  
**Difficulty:** Beginner to Intermediate  
**Purpose:** College lab experiments, self-learning, interview preparation

---

## 🛠 Prerequisites

### Required Software
- Python 3.8 or higher
- pip (Python package manager)

### Required Python Packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests openpyxl joblib
```

### Optional Tools
- Jupyter Notebook (for interactive exploration)
- SQLite Browser (for database visualization)
- Git (for version control)

---

## 📋 Experiments List

### ⚡ Quick Reference Table

| # | Experiment | Focus Area | Difficulty | Execution Time | Key Learning |
|---|------------|------------|------------|----------------|--------------|
| 1 | [Baseline ML Model](#experiment-1-baseline-ml-model) | Machine Learning | ⭐⭐ | ~2 sec | Model training basics |
| 2 | [Label Noise Cleanup](#experiment-2-label-noise-cleanup) | Data Quality | ⭐⭐ | ~3 sec | Noise detection |
| 3 | [ML Pipeline](#experiment-3-ml-pipeline) | MLOps | ⭐⭐⭐ | ~5 sec | End-to-end pipeline |
| 4 | [Data Quality Scorecard](#experiment-4-data-quality-scorecard) | Data Quality | ⭐⭐ | ~3 sec | Quality metrics |
| 5 | [API Data Collection](#experiment-5-api-data-collection) | Data Collection | ⭐⭐ | ~4 sec | API integration |
| 6 | [ETL Pipeline](#experiment-6-etl-pipeline-to-sqlite) | ETL/Databases | ⭐⭐⭐ | ~3 sec | Extract-Transform-Load |
| 7 | [Data Wrangling Toolkit](#experiment-7-data-wrangling-toolkit) | Data Preprocessing | ⭐⭐⭐ | ~10 sec | 8 core preprocessing steps |
| 8 | [Data Enrichment](#experiment-8-data-enrichment-via-mergejoin) | Data Integration | ⭐⭐⭐ | ~5 sec | SQL-like joins in Python |
| 9 | [Complete EDA Program](#experiment-9-complete-eda-program) | Exploratory Analysis | ⭐⭐⭐ | ~7 sec | End-to-end EDA workflow |

---

## 📖 Detailed Experiment Descriptions

### Experiment 1: Baseline ML Model
**File:** `experiment_baseline.py` / `experiment_baseline_complete.py`

**What it does:**
- Trains a basic machine learning model
- Demonstrates train-test split
- Shows model evaluation metrics

**Key Concepts:**
- Supervised learning
- Model training
- Accuracy metrics

**Run:**
```bash
python experiment_baseline_complete.py
```

---

### Experiment 2: Label Noise Cleanup
**File:** `experiment2_label_noise_cleanup.py`

**What it does:**
- Detects noisy/incorrect labels in datasets
- Implements noise detection algorithms
- Cleans data for better model performance

**Key Concepts:**
- Data quality assessment
- Label validation
- Outlier detection in labels

**Run:**
```bash
python experiment2_label_noise_cleanup.py
```

**Output:**
- Cleaned dataset
- Noise detection report

---

### Experiment 3: ML Pipeline
**File:** `experiment3_ml_pipeline.py`

**What it does:**
- Complete end-to-end ML pipeline
- Feature engineering
- Model training and evaluation
- Pipeline persistence

**Key Concepts:**
- Scikit-learn pipelines
- Feature preprocessing
- Model serialization
- Production-ready ML

**Run:**
```bash
python experiment3_ml_pipeline.py
```

**Output:**
- Trained pipeline model (`.joblib`)
- Performance metrics

---

### Experiment 4: Data Quality Scorecard
**File:** `experiment4_data_quality_scorecard.py`

**What it does:**
- Comprehensive data quality assessment
- Calculates quality scores across multiple dimensions
- Generates quality report

**Key Concepts:**
- Data profiling
- Quality dimensions (completeness, accuracy, consistency)
- Data governance

**Run:**
```bash
python experiment4_data_quality_scorecard.py
```

**Output:**
- Quality scorecard report
- Dimension-wise scores

---

### Experiment 5: API Data Collection
**File:** `experiment5_api_data_collection.py`

**What it does:**
- Fetches data from REST APIs
- Handles API responses
- Processes and stores JSON data
- Error handling for API calls

**Key Concepts:**
- REST API integration
- HTTP requests (GET/POST)
- JSON parsing
- Rate limiting
- Error handling

**Run:**
```bash
python experiment5_api_data_collection.py
```

**Output:**
- Collected API data (CSV/JSON)
- API response logs

---

### Experiment 6: ETL Pipeline to SQLite
**File:** `experiment6_etl_pipeline.py`

**What it does:**
- **Extract:** Read data from CSV and JSON files
- **Transform:** Clean, normalize, and process data
- **Load:** Store in SQLite database
- Execute SQL queries from Python

**Key Concepts:**
- ETL process
- Database operations (SQLite)
- SQL queries in Python
- Data transformation
- Schema creation

**Required Files:**
- `students.csv`
- `students.json`

**Run:**
```bash
python experiment6_etl_pipeline.py
```

**Output:**
- `etl_lab.db` - SQLite database
- `etl_pipeline.log` - Detailed logs
- SQL query results in console

**Documentation:** See [ETL_README.md](ETL_README.md)

---

### Experiment 7: Data Wrangling Toolkit
**File:** `experiment7_data_wrangling_toolkit.py`

**What it does:**
Implements **8 core data preprocessing steps**:
1. Handle Missing Values (median/mode imputation)
2. Encode Categorical Data (Label + One-Hot encoding)
3. Feature Scaling (Min-Max + Standardization)
4. Outlier Detection & Removal (IQR method)
5. GroupBy Operations (aggregations)
6. Pivot & Melt (data reshaping)
7. Remove Duplicates
8. Train-Test Split (80-20)

**Key Concepts:**
- Complete data preprocessing workflow
- Feature engineering
- Data validation
- Visualization (boxplots, heatmaps)

**Required Files:**
- `employees.csv`

**Run:**
```bash
python experiment7_data_wrangling_toolkit.py
```

**Output:**
- `employees_cleaned.csv` - Processed dataset
- `outlier_detection.png` - Boxplot visualization
- `correlation_heatmap.png` - Feature correlations

**Documentation:** See [DATA_WRANGLING_README.md](DATA_WRANGLING_README.md)

---

### Experiment 8: Data Enrichment via Merge/Join
**File:** `experiment8_data_enrichment.py`

**What it does:**
- Enrich primary dataset using lookup tables
- Demonstrate **3 join types** (INNER, LEFT, RIGHT)
- Analyze unmatched keys
- Create **4 derived features**
- Generate insights through visualizations

**Key Concepts:**
- Data enrichment
- SQL-like joins in Pandas
- Unmatched key analysis
- Feature engineering
- Data integration

**Required Files:**
- `employees.csv`
- `departments_lookup.csv`

**Run:**
```bash
python experiment8_data_enrichment.py
```

**Output:**
- `employees_enriched.csv` - Final enriched dataset (13 columns)
- `unmatched_employees.csv` - Records without matches
- `data_enrichment_visualizations.png` - Charts

**Documentation:** See [DATA_ENRICHMENT_README.md](DATA_ENRICHMENT_README.md)

---

### Experiment 9: Complete EDA Program
**File:** `experiment9_eda_complete.py`

**What it does:**
Complete **end-to-end Exploratory Data Analysis** workflow:
1. **Data Inspection:** Load and examine raw data
2. **Data Cleaning:** Remove duplicates, handle missing values
3. **Univariate Analysis:** Analyze individual variables
4. **Bivariate/Multivariate Analysis:** Relationships between variables
5. **Outlier Detection:** IQR and Z-score methods
6. **Feature Engineering:** Standardization, encoding, derived features

**Key Concepts:**
- Complete EDA workflow
- Statistical analysis (correlation, distributions)
- Outlier detection (IQR + Z-score)
- Feature transformation
- Data visualization (7 plots)
- Answer research questions with data

**Research Question:**
"Does income depend on age?"

**Run:**
```bash
python experiment9_eda_complete.py
```

**Output:**
- `eda_processed_data.csv` - Final processed dataset (13 records, 11 features)
- `eda_income_distribution.png` - Histogram + boxplot
- `eda_age_distribution.png` - Age distribution
- `eda_age_vs_income_scatter.png` - Scatter plot with trend line
- `eda_correlation_heatmap.png` - Correlation matrix
- `eda_income_by_city.png` - City-wise analysis
- `eda_income_by_gender.png` - Gender-wise analysis
- `eda_outlier_detection.png` - Outlier visualization
- `eda_summary.json` - Statistical summary
- `eda_analysis.log` - Detailed execution logs

**Key Finding:**
- **Strong positive correlation (0.98)** between age and income
- Income increases with age
- 1 outlier detected (60 years, ₹120,000)

---

## 🚀 Quick Start Guide

### Step 1: Clone/Download
```bash
cd "C:\Users\manoh\OneDrive\Desktop\AI"
```

### Step 2: Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests openpyxl joblib
```

### Step 3: Run Any Experiment
```bash
# Example: Run ETL Pipeline
python experiment6_etl_pipeline.py

# Example: Run Data Wrangling
python experiment7_data_wrangling_toolkit.py

# Example: Run Data Enrichment
python experiment8_data_enrichment.py

# Example: Run Complete EDA
python experiment9_eda_complete.py
```

### Step 4: Check Outputs
Look for generated files:
- `.csv` files (processed datasets)
- `.png` files (visualizations)
- `.db` files (databases)
- `.log` files (execution logs)

---

## 📁 Dataset Files

### Input Datasets

| File | Used By | Records | Description |
|------|---------|---------|-------------|
| `data.csv` | Exp 1, 2, 3, 4 | Varies | Generic ML dataset |
| `raw_dataset.csv` | Multiple | Varies | Raw data for processing |
| `students.csv` | Exp 6 | 15 | Student records (CSV) |
| `students.json` | Exp 6 | 10 | Student records (JSON) |
| `employees.csv` | Exp 7, 8 | 53 | Employee records |
| `departments_lookup.csv` | Exp 8 | 7 | Department lookup table |

### Output Datasets

| File | Created By | Description |
|------|------------|-------------|
| `employees_cleaned.csv` | Exp 7 | Processed employee data |
| `employees_enriched.csv` | Exp 8 | Enriched with department info |
| `unmatched_employees.csv` | Exp 8 | Unmatched records |
| `etl_lab.db` | Exp 6 | SQLite database |

---

## 📊 Output Files

### Visualizations

| File | Created By | Type | Description |
|------|------------|------|-------------|
| `outlier_detection.png` | Exp 7 | Boxplot | Before/after outlier removal |
| `correlation_heatmap.png` | Exp 7 | Heatmap | Feature correlations |
| `data_enrichment_visualizations.png` | Exp 8 | Multiple | Employee distribution + salary analysis |

### Logs & Reports

| File | Created By | Description |
|------|------------|-------------|
| `etl_pipeline.log` | Exp 6 | ETL execution logs |
| `metrics.json` | Exp 1, 3 | Model performance metrics |
| `config.json` | Multiple | Configuration settings |

### Models

| File | Created By | Description |
|------|------------|-------------|
| `trained_model.joblib` | Exp 3 | Serialized ML model |

---

## 🎓 Learning Path9 (Complete EDA)
   - Learn exploratory data analysis
   - Understand statistical concepts
   - Visualize data patterns

2. **Next:** Experiment 6 (ETL Pipeline)
   - Learn data extraction, transformation, loading
   - Understand SQL basics
   
3. **Then:** Experiment 7 (Data Wrangling)
   - Master data preprocessing
   - Learn feature engineering
   
5. **Continue:** Experiment 1 (Baseline ML)
   - Basic machine learning
   - Train-test split
   
6. **Advance:** Experiment 3 (ML Pipeline)
   - End-to-end ML workflow
   - Pipeline creation
   
7. **Continue:** Experiment 1 (Baseline ML)
   - Basic machine learning
   - Train-test split
   
5. **Advance:** Experiment 3 (ML Pipeline)
8. **Explore:** Experiment 5 (API Collection)
   - Real-world data sources
   - API integration
   
9  - Advanced data cleaning
   - Quality assessment

### Advanced Track
7. **Explore:** Experiment 5 (API Collection)
   - Real-world data sources
   - API integration
   
8. *Exploratory Data Analysis (Exp 9)
- ✅ Complete EDA workflow
- ✅ Statistical analysis
- ✅ Outlier detection methods
- ✅ Research question answering

### *Integrate:** Combine multiple experiments
   - Build complete data systems
   - Create production pipelines

---

## 💡 Key Takeaways by Experiment

### Data Processing (Exp 6, 7, 8)
- ✅ ETL workflows
- ✅ Data cleaning techniques
- ✅ Feature engineering
- ✅ Data integration patterns

### Machine Learning (Exp 1, 2, 3)
- ✅ Model training
- ✅ Pipeline creation
- ✅ Performance evaluation
- ✅ Model persistence

### Data Quality (Exp 2, 4, 7)
- ✅ Quality assessment
- ✅ Noise detection
- ✅ Validation techniques
- ✅ Profiling methods

### Data Collection (Exp 5)
- ✅ API integration
- ✅ Error handling
- ✅ Data parsing
- ✅ Storage patterns

---

## 🔧 Troubleshooting

### Common Issues

**Issue: Module not found**
```bash
# Solution:
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Issue: File not found error**
```bash
# Solution: Ensure you're in the correct directory
cd "C:\Users\manoh\OneDrive\Desktop\AI"
```

**Issue: Permission denied**
```bash
# Solution: Close any open files (CSV, Excel, DB Browser)
# Or run terminal as administrator
```

**Issue: Encoding errors (Windows)**
```bash
# Already handled in Exp 6, 7, 8 with UTF-8 encoding
```

---

## 📚 Additional Resources

### Documentation Files
- `ETL_README.md` - Complete ETL pipeline guide
- `DATA_WRANGLING_README.md` - Data wrangling detailed docs
- `DATA_ENRICHMENT_README.md` - Data enrichment guide
- `QUICKSTART_WRANGLING.md` - Quick reference for Exp 7
- `QUICKSTART_ENRICHMENT.md` - Quick reference for Exp 8

### External Resources
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Python SQLite Tutorial](https://docs.python.org/3/library/sqlite3.html)
- [Matplotlib Gallery]8,200+

Coverage:
├── Exploratory Analysis: 11%
├── ETL/Data Engineering: 33%
├── Machine Learning: 22%
├── Data Quality: 22%
└── API Integration: 11%

Output Files Generated:
├── CSV: 10+
├── PNG: 10+
├── DB: 1
├── JSON: 2+
└── LOG: 2
Coverage:
├── ETL/Data Engineering: 33%
├── Machine Learning: 33%
├── Data Quality: 22%
└── API Integration: 11%

Output Files Generated:
├── CSV: 8+
├── PNG: 3
├── DB: 1
└── LOG: 1+
```

---

## 🌟 Best Practices Demonstrated

✅ **Modular Code:** Functions for each major operation  
✅ **Error Handling:** Try-except blocks throughout  
✅ **Documentation:** Comprehensive docstrings  
✅ **Logging:** Track execution progress  
✅ **Validation:** Data quality checks  
✅ **Visualization:** Charts for insights  
✅ **Type Safety:** Appropriate data types  
✅ **Performance:** Efficient pandas operations  

---

## 🤝 Contributing

Feel free to:
- Add new experiments
- Improve existing code
- Enhance documentation
- Report issues
- Suggest features

---

## 📄 License

Educational project for college lab experiments and self-learning.

---

## 👨‍💻 Author
Exploratory Data Analysis
python experiment9_eda_complete.py

# Run all ETL/Data Engineering experiments
python experiment6_etl_pipeline.py
python experiment7_data_wrangling_toolkit.py
python experiment8_data_enrichment.py

# Run all ML experiments
python experiment_baseline_complete.py
python experiment3_ml_pipeline.py

# Run data quality experiments
python experiment2_label_noise_cleanup.py
python experiment4_data_quality_scorecard.py

# Run data collection
python experiment5_api_data_collection.py

# View generated files
ls *.csv, *.png, *.db, *.log, *.json
# Run all ML experiments
python experiment_baseline_complete.py
python experiment3_ml_pipeline.py

# Run data quality experiments
python experiment2_label_noise_cleanup.py
python experiment4_data_quality_scorecard.py

# Run data collection
python experiment5_api_data_collection.py

# View generated files
ls *.csv, *.png, *.db, *.log
```

---

**🎉 All experiments are fully functional and ready to use!**

**Happy Learning! 🚀📊🎓**
