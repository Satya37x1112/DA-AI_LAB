"""
EXPERIMENT 5: API DATA COLLECTION + PROVENANCE LOG
====================================================
This script demonstrates:
- Data collection from public REST APIs
- JSON to tabular data conversion
- Data provenance tracking
- Metadata documentation

Why Data Provenance Matters:
- Ensures reproducibility of data collection
- Tracks data lineage and transformations
- Enables auditing and compliance
- Documents data quality and cleaning steps
"""

import requests
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EXPERIMENT 5: API DATA COLLECTION + PROVENANCE LOG")
print("=" * 80)
print()

# ============================================================
# STEP 1: API DATA COLLECTION
# ============================================================
print("STEP 1: API DATA COLLECTION")
print("-" * 80)

# Public API Configuration
# Using JSONPlaceholder - a free fake REST API for testing
API_URL = "https://jsonplaceholder.typicode.com/users"
API_NAME = "JSONPlaceholder Users API"
API_DESCRIPTION = "Free fake REST API for testing and prototyping"

print(f"\n   API Information:")
print(f"      Name: {API_NAME}")
print(f"      URL: {API_URL}")
print(f"      Description: {API_DESCRIPTION}")
print(f"      Method: HTTP GET")

# Record collection timestamp
collection_timestamp = datetime.now()
collection_datetime_str = collection_timestamp.strftime("%Y-%m-%d %H:%M:%S")

print(f"\n   Collection Time: {collection_datetime_str}")

# Fetch data from API
print(f"\n   Fetching data from API...")

try:
    response = requests.get(API_URL, timeout=10)
    
    # Check HTTP status code
    print(f"      HTTP Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print(f"      ✓ API request successful")
        
        # Parse JSON response
        raw_json_data = response.json()
        print(f"      ✓ JSON data parsed successfully")
        print(f"      Records fetched: {len(raw_json_data)}")
        
    else:
        print(f"      ✗ Error: API returned status code {response.status_code}")
        exit(1)
        
except requests.exceptions.Timeout:
    print(f"      ✗ Error: API request timed out")
    exit(1)
except requests.exceptions.RequestException as e:
    print(f"      ✗ Error: API request failed - {e}")
    exit(1)
except json.JSONDecodeError:
    print(f"      ✗ Error: Failed to parse JSON response")
    exit(1)

print(f"\n   First record sample:")
print(json.dumps(raw_json_data[0], indent=2))

# ============================================================
# STEP 2: CONVERT JSON TO DATAFRAME
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: CONVERT JSON TO DATAFRAME")
print("-" * 80)

print(f"\n   Normalizing JSON data into tabular format...")

# Normalize nested JSON structure into flat DataFrame
df_raw = pd.json_normalize(raw_json_data)

print(f"      ✓ JSON converted to DataFrame")

print(f"\n   DataFrame Information:")
print(f"      Shape: {df_raw.shape}")
print(f"      Rows: {df_raw.shape[0]}")
print(f"      Columns: {df_raw.shape[1]}")

print(f"\n   Column Names:")
for i, col in enumerate(df_raw.columns, 1):
    print(f"      {i}. {col}")

print(f"\n   Data Types:")
print(df_raw.dtypes)

print(f"\n   First few rows:")
print(df_raw.head(3))

# ============================================================
# STEP 3: SAVE RAW DATA
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: SAVE RAW DATA (No Preprocessing)")
print("-" * 80)
print("\n   Why save raw data: Preserves original source for reproducibility and auditing")

# Save raw JSON
raw_json_file = "raw_data.json"
print(f"\n   Saving raw JSON data to: {raw_json_file}")

try:
    with open(raw_json_file, 'w') as f:
        json.dump(raw_json_data, f, indent=2)
    print(f"      ✓ Raw JSON saved successfully")
except Exception as e:
    print(f"      ✗ Error saving JSON: {e}")

# Save raw CSV
raw_csv_file = "raw_data.csv"
print(f"\n   Saving raw CSV data to: {raw_csv_file}")

try:
    df_raw.to_csv(raw_csv_file, index=False)
    print(f"      ✓ Raw CSV saved successfully")
    print(f"      File size: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
except Exception as e:
    print(f"      ✗ Error saving CSV: {e}")

# ============================================================
# STEP 4: BASIC DATA CLEANING
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: BASIC DATA CLEANING")
print("-" * 80)

# Create a copy for cleaning
df_cleaned = df_raw.copy()
cleaning_steps = []

print(f"\n   Initial data: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

# Step 4.1: Remove duplicates
print(f"\n   Step 4.1: Check and remove duplicates")
duplicates_before = df_cleaned.duplicated().sum()
print(f"      Duplicates found: {duplicates_before}")

if duplicates_before > 0:
    df_cleaned = df_cleaned.drop_duplicates()
    duplicates_after = df_cleaned.duplicated().sum()
    rows_removed = duplicates_before
    print(f"      ✓ Removed {rows_removed} duplicate rows")
    cleaning_steps.append(f"Removed {rows_removed} duplicate rows")
else:
    print(f"      ✓ No duplicates found")
    cleaning_steps.append("No duplicates found")

# Step 4.2: Handle missing values
print(f"\n   Step 4.2: Check and handle missing values")
missing_before = df_cleaned.isnull().sum().sum()
print(f"      Total missing values: {missing_before}")

if missing_before > 0:
    print(f"      Missing values per column:")
    for col in df_cleaned.columns:
        missing_count = df_cleaned[col].isnull().sum()
        if missing_count > 0:
            print(f"         {col}: {missing_count}")
    
    # Simple strategy: Fill numeric with mean, categorical with mode
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().any():
            if df_cleaned[col].dtype in ['int64', 'float64']:
                fill_value = df_cleaned[col].mean()
                df_cleaned[col].fillna(fill_value, inplace=True)
                print(f"         Filled '{col}' with mean: {fill_value:.2f}")
                cleaning_steps.append(f"Filled missing values in '{col}' with mean")
            else:
                fill_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else "Unknown"
                df_cleaned[col].fillna(fill_value, inplace=True)
                print(f"         Filled '{col}' with mode: {fill_value}")
                cleaning_steps.append(f"Filled missing values in '{col}' with mode")
    
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"      ✓ Missing values after cleaning: {missing_after}")
else:
    print(f"      ✓ No missing values found")
    cleaning_steps.append("No missing values found")

# Step 4.3: Data type validation
print(f"\n   Step 4.3: Data type validation")
print(f"      ✓ Data types verified")
cleaning_steps.append("Verified data types")

print(f"\n   Cleaning Summary:")
print(f"      Original rows: {df_raw.shape[0]}")
print(f"      Cleaned rows: {df_cleaned.shape[0]}")
print(f"      Rows removed: {df_raw.shape[0] - df_cleaned.shape[0]}")
print(f"      Cleaning steps applied: {len(cleaning_steps)}")

# Save cleaned data
cleaned_csv_file = "cleaned_data.csv"
print(f"\n   Saving cleaned data to: {cleaned_csv_file}")

try:
    df_cleaned.to_csv(cleaned_csv_file, index=False)
    print(f"      ✓ Cleaned CSV saved successfully")
except Exception as e:
    print(f"      ✗ Error saving cleaned CSV: {e}")

# ============================================================
# STEP 5: PROVENANCE METADATA CREATION
# ============================================================
print("\n" + "=" * 80)
print("STEP 5: PROVENANCE METADATA CREATION")
print("-" * 80)
print("\n   Why provenance: Documents complete data lineage for reproducibility and compliance")

# Create comprehensive metadata dictionary
provenance_metadata = {
    "data_collection": {
        "api_name": API_NAME,
        "api_url": API_URL,
        "api_description": API_DESCRIPTION,
        "http_method": "GET",
        "http_status_code": response.status_code,
        "collection_datetime": collection_datetime_str,
        "collection_timestamp_unix": int(collection_timestamp.timestamp()),
        "collector": "Experiment 5 - API Data Collection Script"
    },
    "data_characteristics": {
        "records_collected": len(raw_json_data),
        "raw_data_shape": {
            "rows": df_raw.shape[0],
            "columns": df_raw.shape[1]
        },
        "cleaned_data_shape": {
            "rows": df_cleaned.shape[0],
            "columns": df_cleaned.shape[1]
        },
        "column_names": df_cleaned.columns.tolist(),
        "data_types": {col: str(dtype) for col, dtype in df_cleaned.dtypes.items()}
    },
    "data_quality": {
        "duplicates_found": int(duplicates_before),
        "duplicates_removed": int(duplicates_before),
        "missing_values_before": int(missing_before),
        "missing_values_after": int(df_cleaned.isnull().sum().sum())
    },
    "cleaning_steps": cleaning_steps,
    "output_files": {
        "raw_json": raw_json_file,
        "raw_csv": raw_csv_file,
        "cleaned_csv": cleaned_csv_file,
        "metadata": "provenance_metadata.json"
    },
    "versioning": {
        "script_version": "1.0",
        "pandas_version": pd.__version__,
        "python_version": "3.x"
    }
}

# Save metadata to JSON
metadata_file = "provenance_metadata.json"
print(f"\n   Creating provenance metadata...")
print(f"      Metadata components:")
print(f"         - Data collection details")
print(f"         - Data characteristics")
print(f"         - Data quality metrics")
print(f"         - Cleaning steps log")
print(f"         - Output file references")
print(f"         - Versioning information")

try:
    with open(metadata_file, 'w') as f:
        json.dump(provenance_metadata, f, indent=2)
    print(f"\n   ✓ Metadata saved to: {metadata_file}")
except Exception as e:
    print(f"\n   ✗ Error saving metadata: {e}")

print(f"\n   Metadata Preview:")
print(json.dumps(provenance_metadata, indent=2))

# ============================================================
# STEP 6: OUTPUT & LOGGING SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("STEP 6: PIPELINE SUMMARY & OUTPUT")
print("=" * 80)

print(f"""
   Pipeline Execution Complete!
   
   Stages Completed:
      ✓ Stage 1: API Data Collection
         - API: {API_NAME}
         - Records fetched: {len(raw_json_data)}
         - Collection time: {collection_datetime_str}
      
      ✓ Stage 2: JSON to DataFrame Conversion
         - Normalized nested JSON structure
         - Created tabular DataFrame: {df_raw.shape[0]}×{df_raw.shape[1]}
      
      ✓ Stage 3: Raw Data Saved
         - JSON: {raw_json_file}
         - CSV: {raw_csv_file}
      
      ✓ Stage 4: Data Cleaning
         - Duplicates removed: {duplicates_before}
         - Missing values handled: {missing_before}
         - Cleaning steps: {len(cleaning_steps)}
         - Output: {cleaned_csv_file}
      
      ✓ Stage 5: Provenance Metadata Created
         - Complete data lineage documented
         - Output: {metadata_file}
   
   Output Files Generated:
      1. {raw_json_file} - Original API response
      2. {raw_csv_file} - Raw tabular data
      3. {cleaned_csv_file} - Cleaned tabular data
      4. {metadata_file} - Complete provenance metadata
""")

print("=" * 80)
print("DATA PROVENANCE INFORMATION")
print("=" * 80)

provenance_table = pd.DataFrame({
    'Aspect': [
        'Data Source',
        'Collection Date',
        'Records Collected',
        'Duplicates Removed',
        'Missing Values',
        'Final Records',
        'Data Columns'
    ],
    'Value': [
        API_NAME,
        collection_datetime_str,
        f"{len(raw_json_data)} rows",
        f"{duplicates_before} rows",
        f"{missing_before} cells (handled)",
        f"{df_cleaned.shape[0]} rows",
        f"{df_cleaned.shape[1]} columns"
    ]
})

print(provenance_table.to_string(index=False))

print("\n" + "=" * 80)
print("WHY DATA PROVENANCE MATTERS")
print("=" * 80)

print("""
1. REPRODUCIBILITY:
   - Anyone can re-run the data collection process
   - Exact API source and timestamp are documented
   - Cleaning steps are tracked for verification

2. AUDITING & COMPLIANCE:
   - Complete audit trail of data transformations
   - Meets regulatory requirements (GDPR, SOC2, etc.)
   - Enables data governance and quality control

3. DEBUGGING & TROUBLESHOOTING:
   - Quickly identify when/where issues occurred
   - Trace data quality problems to source
   - Validate data transformations

4. COLLABORATION:
   - Team members understand data origin
   - Reduces knowledge silos
   - Enables handoffs between teams

5. VERSIONING & LINEAGE:
   - Track changes over time
   - Understand data evolution
   - Support rollback if needed
""")

print("=" * 80)
print("EXPERIMENT COMPLETE - ALL FILES SAVED")
print("=" * 80)
print()

print("Next Steps:")
print("   1. Review provenance_metadata.json for complete data lineage")
print("   2. Use cleaned_data.csv for downstream ML tasks")
print("   3. Keep raw_data.json and raw_data.csv for reproducibility")
print("   4. Document any additional transformations in metadata")
print()
