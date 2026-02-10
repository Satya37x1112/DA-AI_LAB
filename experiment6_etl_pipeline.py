"""
ETL Pipeline to SQLite Database
================================
Experiment Title: ETL Pipeline to SQLite

Aim: Implement Extract → Transform → Load process using Python

Author: Data Engineering Lab
Date: February 10, 2026

This program demonstrates a complete ETL pipeline that:
1. Extracts data from CSV and JSON files
2. Transforms data through cleaning and normalization
3. Loads processed data into SQLite database
4. Executes SQL queries for data analysis
"""

import pandas as pd
import sqlite3
import logging
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION & LOGGING SETUP
# ============================================================================

# Configure logging for ETL pipeline tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Database configuration
DB_NAME = 'etl_lab.db'
TABLE_NAME = 'students'

# File paths
CSV_FILE = 'students.csv'
JSON_FILE = 'students.json'


# ============================================================================
# PHASE 1: EXTRACT
# ============================================================================

def extract_data():
    """
    Extract data from CSV and JSON files.
    
    Returns:
        pd.DataFrame: Combined dataframe from both sources
        
    Raises:
        FileNotFoundError: If required files are not found
        Exception: For other extraction errors
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: EXTRACT - Starting data extraction")
    logger.info("=" * 60)
    
    dataframes = []
    
    try:
        # Extract from CSV
        if os.path.exists(CSV_FILE):
            logger.info(f"Reading data from {CSV_FILE}...")
            df_csv = pd.read_csv(CSV_FILE)
            logger.info(f"✓ CSV loaded: {len(df_csv)} records")
            dataframes.append(df_csv)
        else:
            logger.warning(f"⚠ {CSV_FILE} not found, skipping...")
            
        # Extract from JSON
        if os.path.exists(JSON_FILE):
            logger.info(f"Reading data from {JSON_FILE}...")
            df_json = pd.read_json(JSON_FILE)
            logger.info(f"✓ JSON loaded: {len(df_json)} records")
            dataframes.append(df_json)
        else:
            logger.warning(f"⚠ {JSON_FILE} not found, skipping...")
        
        # Check if any data was loaded
        if not dataframes:
            raise FileNotFoundError(
                f"No data files found. Please ensure {CSV_FILE} and/or {JSON_FILE} exist."
            )
        
        # Combine dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        logger.info(f"✓ Total records extracted: {len(combined_df)}")
        logger.info(f"✓ Columns found: {list(combined_df.columns)}")
        
        print("\n" + "=" * 60)
        print("✓ EXTRACT PHASE COMPLETED")
        print("=" * 60)
        print(f"Total records extracted: {len(combined_df)}\n")
        
        return combined_df
        
    except FileNotFoundError as e:
        logger.error(f"✗ File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Error during extraction: {e}")
        raise


# ============================================================================
# PHASE 2: TRANSFORM
# ============================================================================

def transform_data(df):
    """
    Transform and clean the extracted data.
    
    Transformations include:
    - Remove duplicates
    - Handle missing values
    - Convert data types
    - Standardize text fields
    - Rename columns to SQL-friendly names
    - Calculate grade based on marks
    
    Args:
        df (pd.DataFrame): Raw extracted dataframe
        
    Returns:
        pd.DataFrame: Cleaned and transformed dataframe
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: TRANSFORM - Starting data transformation")
    logger.info("=" * 60)
    
    try:
        # Create a copy to avoid modifying original
        df_transformed = df.copy()
        
        # 1. Remove duplicate records
        initial_count = len(df_transformed)
        df_transformed = df_transformed.drop_duplicates()
        duplicates_removed = initial_count - len(df_transformed)
        logger.info(f"✓ Duplicates removed: {duplicates_removed}")
        
        # 2. Rename columns to SQL-friendly names (lowercase, underscore)
        column_mapping = {
            'Name': 'name',
            'Age': 'age',
            'Marks': 'marks',
            'City': 'city'
        }
        df_transformed = df_transformed.rename(columns=column_mapping)
        logger.info(f"✓ Columns renamed to SQL-friendly format")
        
        # 3. Handle missing values
        missing_before = df_transformed.isnull().sum().sum()
        
        # Fill missing names with 'Unknown'
        if 'name' in df_transformed.columns:
            df_transformed['name'] = df_transformed['name'].fillna('Unknown')
        
        # Fill missing cities with 'Not Specified'
        if 'city' in df_transformed.columns:
            df_transformed['city'] = df_transformed['city'].fillna('Not Specified')
        
        # Drop rows with missing age or marks (critical fields)
        df_transformed = df_transformed.dropna(subset=['age', 'marks'])
        
        missing_after = df_transformed.isnull().sum().sum()
        logger.info(f"✓ Missing values handled: {missing_before} → {missing_after}")
        
        # 4. Convert data types
        df_transformed['age'] = df_transformed['age'].astype(int)
        df_transformed['marks'] = df_transformed['marks'].astype(float)
        df_transformed['name'] = df_transformed['name'].astype(str)
        df_transformed['city'] = df_transformed['city'].astype(str)
        logger.info(f"✓ Data types converted (age→int, marks→float, text→str)")
        
        # 5. Standardize text fields
        # Strip spaces and convert to proper case
        df_transformed['name'] = df_transformed['name'].str.strip().str.title()
        df_transformed['city'] = df_transformed['city'].str.strip().str.title()
        logger.info(f"✓ Text fields standardized (stripped, proper case)")
        
        # 6. Add grade column based on marks
        def calculate_grade(marks):
            """Calculate grade based on marks."""
            if marks >= 80:
                return 'A'
            elif marks >= 60:
                return 'B'
            elif marks >= 40:
                return 'C'
            else:
                return 'F'
        
        df_transformed['grade'] = df_transformed['marks'].apply(calculate_grade)
        logger.info(f"✓ Grade column added based on marks")
        
        # 7. Sort by marks (descending)
        df_transformed = df_transformed.sort_values('marks', ascending=False).reset_index(drop=True)
        
        # Display transformation summary
        logger.info(f"\nTransformation Summary:")
        logger.info(f"  Final record count: {len(df_transformed)}")
        logger.info(f"  Columns: {list(df_transformed.columns)}")
        logger.info(f"  Data types: {dict(df_transformed.dtypes)}")
        
        print("\n" + "=" * 60)
        print("✓ TRANSFORM PHASE COMPLETED")
        print("=" * 60)
        print(f"Records after transformation: {len(df_transformed)}")
        print(f"\nSample of transformed data:")
        print(df_transformed.head(3).to_string(index=False))
        print()
        
        return df_transformed
        
    except Exception as e:
        logger.error(f"✗ Error during transformation: {e}")
        raise


# ============================================================================
# PHASE 3: LOAD
# ============================================================================

def load_data(df):
    """
    Load transformed data into SQLite database.
    
    Args:
        df (pd.DataFrame): Transformed dataframe to load
        
    Returns:
        sqlite3.Connection: Database connection object
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: LOAD - Loading data into SQLite database")
    logger.info("=" * 60)
    
    try:
        # Connect to SQLite database (creates if doesn't exist)
        conn = sqlite3.connect(DB_NAME)
        logger.info(f"✓ Connected to database: {DB_NAME}")
        
        # Load dataframe to SQLite table (replace if exists)
        df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        logger.info(f"✓ Data loaded into table: {TABLE_NAME}")
        logger.info(f"✓ Records inserted: {len(df)}")
        
        # Verify the load
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        count = cursor.fetchone()[0]
        logger.info(f"✓ Verification: {count} records in database")
        
        print("\n" + "=" * 60)
        print("✓ LOAD PHASE COMPLETED")
        print("=" * 60)
        print(f"Database: {DB_NAME}")
        print(f"Table: {TABLE_NAME}")
        print(f"Records loaded: {count}\n")
        
        return conn
        
    except Exception as e:
        logger.error(f"✗ Error during load: {e}")
        raise


# ============================================================================
# DATABASE SCHEMA DISPLAY (BONUS)
# ============================================================================

def show_table_schema(conn):
    """
    Display the database table schema.
    
    Args:
        conn (sqlite3.Connection): Database connection
    """
    logger.info("\n" + "=" * 60)
    logger.info("BONUS: Displaying Table Schema")
    logger.info("=" * 60)
    
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
        schema = cursor.fetchall()
        
        print("\n" + "=" * 60)
        print(f"TABLE SCHEMA: {TABLE_NAME}")
        print("=" * 60)
        print(f"{'Column':<15} {'Type':<15} {'Null':<10} {'Key':<10}")
        print("-" * 60)
        
        for col in schema:
            col_id, name, dtype, not_null, default, pk = col
            null_allowed = "NO" if not_null else "YES"
            is_key = "PRIMARY" if pk else ""
            print(f"{name:<15} {dtype:<15} {null_allowed:<10} {is_key:<10}")
        
        print("=" * 60 + "\n")
        
    except Exception as e:
        logger.error(f"✗ Error displaying schema: {e}")


# ============================================================================
# SQL QUERIES & ANALYSIS
# ============================================================================

def run_queries(conn):
    """
    Execute SQL queries for data analysis.
    
    Args:
        conn (sqlite3.Connection): Database connection
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXECUTING SQL QUERIES FOR DATA ANALYSIS")
    logger.info("=" * 60)
    
    try:
        cursor = conn.cursor()
        
        # Query 1: Show all records
        print("\n" + "=" * 60)
        print("QUERY 1: All Student Records")
        print("=" * 60)
        query1 = f"SELECT * FROM {TABLE_NAME}"
        df_all = pd.read_sql_query(query1, conn)
        print(df_all.to_string(index=False))
        logger.info(f"✓ Query 1 executed: {len(df_all)} records")
        
        # Query 2: Students scoring above 75
        print("\n" + "=" * 60)
        print("QUERY 2: Students Scoring Above 75")
        print("=" * 60)
        query2 = f"SELECT name, marks, grade FROM {TABLE_NAME} WHERE marks > 75 ORDER BY marks DESC"
        df_high_scorers = pd.read_sql_query(query2, conn)
        if len(df_high_scorers) > 0:
            print(df_high_scorers.to_string(index=False))
        else:
            print("No students scoring above 75.")
        logger.info(f"✓ Query 2 executed: {len(df_high_scorers)} students above 75")
        
        # Query 3: Average marks
        print("\n" + "=" * 60)
        print("QUERY 3: Average Marks of All Students")
        print("=" * 60)
        query3 = f"SELECT ROUND(AVG(marks), 2) as average_marks FROM {TABLE_NAME}"
        cursor.execute(query3)
        avg_marks = cursor.fetchone()[0]
        print(f"Average Marks: {avg_marks}")
        logger.info(f"✓ Query 3 executed: Average marks = {avg_marks}")
        
        # Query 4: Count students grade-wise
        print("\n" + "=" * 60)
        print("QUERY 4: Student Count by Grade")
        print("=" * 60)
        query4 = f"SELECT grade, COUNT(*) as student_count FROM {TABLE_NAME} GROUP BY grade ORDER BY grade"
        df_grade_count = pd.read_sql_query(query4, conn)
        print(df_grade_count.to_string(index=False))
        logger.info(f"✓ Query 4 executed: Grade distribution calculated")
        
        # Query 5: Top 5 highest scorers
        print("\n" + "=" * 60)
        print("QUERY 5: Top 5 Highest Scorers")
        print("=" * 60)
        query5 = f"SELECT name, age, marks, grade, city FROM {TABLE_NAME} ORDER BY marks DESC LIMIT 5"
        df_top5 = pd.read_sql_query(query5, conn)
        print(df_top5.to_string(index=False))
        logger.info(f"✓ Query 5 executed: Top 5 scorers displayed")
        
        print("\n" + "=" * 60)
        print("✓ ALL QUERIES COMPLETED SUCCESSFULLY")
        print("=" * 60 + "\n")
        
    except Exception as e:
        logger.error(f"✗ Error executing queries: {e}")
        raise


# ============================================================================
# MAIN ETL PIPELINE
# ============================================================================

def main():
    """
    Main function to orchestrate the complete ETL pipeline.
    """
    print("\n" + "=" * 60)
    print("ETL PIPELINE TO SQLite DATABASE".center(60))
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")
    
    logger.info("ETL Pipeline started")
    
    conn = None
    
    try:
        # PHASE 1: EXTRACT
        raw_data = extract_data()
        
        # PHASE 2: TRANSFORM
        transformed_data = transform_data(raw_data)
        
        # PHASE 3: LOAD
        conn = load_data(transformed_data)
        
        # BONUS: Show table schema
        show_table_schema(conn)
        
        # PHASE 4: SQL QUERIES
        run_queries(conn)
        
        # Success message
        print("\n" + "=" * 60)
        print("ETL PIPELINE COMPLETED SUCCESSFULLY! ✓".center(60))
        print("=" * 60)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Database file: {DB_NAME}")
        print(f"Log file: etl_pipeline.log")
        print("=" * 60 + "\n")
        
        logger.info("✓ ETL Pipeline completed successfully")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: ETL PIPELINE FAILED ✗".center(60))
        print("=" * 60)
        print(f"Error: {str(e)}")
        print("Check etl_pipeline.log for details.")
        print("=" * 60 + "\n")
        logger.error(f"✗ ETL Pipeline failed: {e}")
        
    finally:
        # Close database connection
        if conn:
            conn.close()
            logger.info("Database connection closed")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
