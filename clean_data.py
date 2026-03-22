# clean_data.py
import pandas as pd
import numpy as np

def clean_traffic_data():
    print("Reading raw data...")
    # Read the file - assuming first row is header
    df = pd.read_csv("raw_traffic.csv", low_memory=False)
    
    print(f"Original rows: {len(df):,}")
    print("\nOriginal columns:", df.columns.tolist())
    
    # Standardize column names: lower case, replace spaces and dots with underscore
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r'\s+', '_', regex=True)
        .str.replace('.', '_', regex=False)
    )
    
    print("\nNormalized columns:", df.columns.tolist())
    
    # ── Try to find and convert date/time if they exist ───────────────────────
    # In your sample, there is NO date/time column → we skip or create dummy if needed
    date_col = None
    time_col = None
    
    possible_date_cols = [c for c in df.columns if 'date' in c]
    possible_time_cols = [c for c in df.columns if 'time' in c]
    
    if possible_date_cols:
        date_col = possible_date_cols[0]
        print(f"Using date column: {date_col}")
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        print("Warning: No date column found → datetime features will be skipped")
    
    if possible_time_cols:
        time_col = possible_time_cols[0]
        print(f"Using time column: {time_col}")
        # Very simple time parsing - adjust if format is unusual
        def parse_time(val):
            if pd.isna(val):
                return pd.NaT
            try:
                return pd.to_datetime(str(val).strip(), format='%H:%M:%S', errors='coerce').time()
            except:
                return pd.NaT
        
        df[time_col] = df[time_col].apply(parse_time)
    
    # If we have both date and time → create datetime
    if date_col and time_col:
        df['datetime'] = pd.to_datetime(
            df[date_col].astype(str) + ' ' + df[time_col].astype(str),
            errors='coerce'
        )
        df = df.dropna(subset=[date_col])  # remove invalid dates if any
    elif date_col:
        df['datetime'] = df[date_col]
    else:
        df['datetime'] = pd.NaT
    
    # ── Boolean columns ────────────────────────────────────────────────────────
    bool_map = {
        'Yes': True, 'YES': True, 'yes': True, 'Y': True,
        'No': False, 'NO': False, 'no': False, 'N': False
    }
    
    bool_columns = [
        'belts', 'personal_injury', 'property_damage', 'commercial_license',
        'commercial_vehicle', 'contributed_to_accident'
        # add others if they appear later: 'fatal', 'hazmat', etc.
    ]
    
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].map(bool_map).fillna(False)
        else:
            print(f"Note: Boolean column '{col}' not found")
    
    # ── Feature engineering ────────────────────────────────────────────────────
    if 'datetime' in df.columns and df['datetime'].notna().any():
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['month'] = df['datetime'].dt.month
        df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
    else:
        print("No valid datetime → skipping time-based features")
    
    # Violation grouping based on Description
    if 'description' in df.columns:
        def categorize_violation(text):
            if pd.isna(text):
                return "Unknown"
            t = str(text).lower()
            if 'speed' in t or 'exceeding' in t:
                return "Speeding"
            if 'red light' in t or 'traffic signal' in t:
                return "Red Light / Signal"
            if 'registration' in t or 'expired' in t and 'plate' in t:
                return "Registration / Plate"
            if 'license' in t or 'suspended' in t:
                return "License / Suspended"
            if 'seatbelt' in t or 'belt' in t or 'restrained' in t:
                return "Seatbelt"
            if 'stop sign' in t:
                return "Stop Sign"
            if 'alcohol' in t or 'influence' in t:
                return "DUI / Alcohol"
            return "Other"
        
        df['violation_group'] = df['description'].apply(categorize_violation)
    
    # ── Save cleaned version ──────────────────────────────────────────────────
    output_file = "cleaned_traffic.parquet"
    df.to_parquet(output_file, index=False, engine="pyarrow")
    
    print(f"\nSaved cleaned file → {output_file}")
    print(f"Final rows: {len(df):,}")
    print("Final columns:", df.columns.tolist()[:25])  # show first 25
    
    return df


if __name__ == "__main__":
    clean_traffic_data()