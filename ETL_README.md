# ETL Pipeline to SQLite - Lab Experiment

## 📋 Experiment Overview

**Title:** ETL Pipeline to SQLite Database  
**Aim:** Implement Extract → Transform → Load process using Python  
**Tech Stack:** Python, Pandas, SQLite3

---

## 🎯 Learning Objectives

1. Understand the ETL (Extract-Transform-Load) process
2. Work with multiple data formats (CSV, JSON)
3. Perform data cleaning and transformation
4. Load data into a relational database
5. Execute SQL queries from Python

---

## 🔧 Installation

### Required Python Packages

```bash
pip install pandas
```

**Note:** `sqlite3` is included in Python's standard library, so no additional installation is needed.

---

## 📁 Project Files

```
AI/
├── experiment6_etl_pipeline.py    # Main ETL program
├── students.csv                   # CSV data source
├── students.json                  # JSON data source
├── etl_lab.db                    # SQLite database (created after running)
├── etl_pipeline.log              # Log file (created after running)
└── ETL_README.md                 # This file
```

---

## 🚀 How to Run

1. **Ensure all files are in the same directory:**
   - `experiment6_etl_pipeline.py`
   - `students.csv`
   - `students.json`

2. **Run the program:**
   ```bash
   python experiment6_etl_pipeline.py
   ```

3. **Check outputs:**
   - Console output showing each ETL phase
   - `etl_lab.db` - SQLite database file
   - `etl_pipeline.log` - Detailed log file

---

## 📊 Sample Data Structure

### students.csv
```csv
Name,Age,Marks,City
Rahul Sharma,21,85.5,Mumbai
Priya Patel,20,92.0,Delhi
...
```

### students.json
```json
[
    {
        "Name": "Sanjay Desai",
        "Age": 22,
        "Marks": 76.5,
        "City": "Nagpur"
    },
    ...
]
```

---

## 🔄 ETL Process Flow

### 1️⃣ EXTRACT Phase
- Read data from `students.csv`
- Read data from `students.json`
- Combine both datasets
- Handle missing files gracefully

### 2️⃣ TRANSFORM Phase
- **Remove duplicates**
- **Handle missing values:**
  - Fill empty names with "Unknown"
  - Fill empty cities with "Not Specified"
  - Drop rows with missing age/marks
- **Convert data types:**
  - Age → Integer
  - Marks → Float
  - Name/City → String
- **Standardize text:**
  - Strip whitespace
  - Convert to proper case
- **Rename columns** to SQL-friendly format (lowercase, underscores)
- **Add grade column:**
  - Marks ≥ 80 → **A**
  - 60-79 → **B**
  - 40-59 → **C**
  - < 40 → **F**

### 3️⃣ LOAD Phase
- Connect to SQLite database (`etl_lab.db`)
- Create/replace `students` table
- Insert transformed data
- Verify record count

### 4️⃣ QUERY Phase
Execute 5 SQL queries:
1. Show all records
2. Students scoring above 75
3. Average marks of all students
4. Student count by grade
5. Top 5 highest scorers

---

## 📈 Expected Output

### Console Output Structure

```
============================================================
                ETL PIPELINE TO SQLite DATABASE
============================================================

============================================================
✓ EXTRACT PHASE COMPLETED
============================================================
Total records extracted: 25

============================================================
✓ TRANSFORM PHASE COMPLETED
============================================================
Records after transformation: 24

Sample of transformed data:
...

============================================================
✓ LOAD PHASE COMPLETED
============================================================
Database: etl_lab.db
Table: students
Records loaded: 24

============================================================
TABLE SCHEMA: students
============================================================
Column          Type            Null       Key
------------------------------------------------------------
name            TEXT            YES
age             INTEGER         YES
marks           REAL            YES
city            TEXT            YES
grade           TEXT            YES
============================================================

============================================================
QUERY 1: All Student Records
============================================================
...

============================================================
QUERY 2: Students Scoring Above 75
============================================================
...

[Additional queries...]

============================================================
           ETL PIPELINE COMPLETED SUCCESSFULLY! ✓
============================================================
```

---

## 🧪 Testing the Database

You can manually query the database using SQLite command line or any SQLite browser:

```bash
sqlite3 etl_lab.db

sqlite> SELECT * FROM students LIMIT 5;
sqlite> SELECT grade, COUNT(*) FROM students GROUP BY grade;
sqlite> .quit
```

---

## 📝 Code Structure

```python
# Main Functions:
- extract_data()      # Phase 1: Extract from CSV/JSON
- transform_data()    # Phase 2: Clean and transform
- load_data()         # Phase 3: Load to SQLite
- run_queries()       # Phase 4: Execute SQL queries
- show_table_schema() # Bonus: Display schema
- main()              # Orchestrate entire pipeline
```

---

## ✨ Features

✅ Modular design with separate functions  
✅ Comprehensive error handling  
✅ Logging to file and console  
✅ Detailed docstrings  
✅ SQL-friendly column naming  
✅ Data validation and cleaning  
✅ Grade calculation logic  
✅ Database schema display  
✅ Multiple SQL query examples  
✅ Beginner-friendly comments  

---

## 🎓 Learning Points

1. **ETL Concepts:** Understanding data pipeline stages
2. **Data Formats:** Working with CSV and JSON
3. **Pandas Operations:** Data manipulation and cleaning
4. **SQLite Integration:** Database creation and operations
5. **Error Handling:** Graceful failure management
6. **Logging:** Tracking process execution
7. **SQL Queries:** Data analysis and retrieval
8. **Code Organization:** Modular programming practices

---

## 🐛 Troubleshooting

**Error: File not found**
- Ensure `students.csv` and `students.json` are in the same directory as the Python script

**Error: Module not found**
- Run: `pip install pandas`

**Error: Database locked**
- Close any SQLite browser/viewer accessing `etl_lab.db`

**Error: Permission denied**
- Ensure you have write permissions in the directory

---

## 📚 Additional Resources

- [Pandas Documentation](https://pandas.pydata.org/)
- [SQLite Python Tutorial](https://docs.python.org/3/library/sqlite3.html)
- [ETL Best Practices](https://en.wikipedia.org/wiki/Extract,_transform,_load)

---

## 👨‍💻 Author

Data Engineering Lab Experiment  
Date: February 10, 2026

---

## 📄 License

This is an educational project for college lab experiments.

---

**Happy Learning! 🚀**
