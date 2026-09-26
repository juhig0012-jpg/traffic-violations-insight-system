# cleans + enriches the raw traffic violations export (Montgomery County style)
# run: python clean_data.py -> spits out cleaned_traffic.parquet
#
# note: raw_traffic.csv doesn't have stop date/time, agency/location, geo, or
# search columns. they're in the full public dataset but just not in this
# trimmed export - see README for what that means for the dashboard

import sys

import numpy as np
import pandas as pd

RAW_FILE = "raw_traffic.csv"
OUTPUT_FILE = "cleaned_traffic.parquet"

# yes/no cols in this export. Fatal/HAZMAT/Alcohol/Work Zone flags exist in the
# full dataset but never made it into raw_traffic.csv
BOOLEAN_COLUMNS = [
    "belts",
    "personal_injury",
    "property_damage",
    "commercial_license",
    "commercial_vehicle",
    "contributed_to_accident",
]

STATE_COLUMNS = ["state", "driver_state", "dl_state"]

# XX = unknown catch-all, US = data entry mistake basically. leaving Canadian
# provinces (ON, QC, MB...) alone since those are legit out-of-country drivers
INVALID_STATE_CODES = {"XX", "US"}

# not exhaustive - just the abbreviations that kept showing up when I checked value_counts
MAKE_ALIASES = {
    "TOYT": "TOYOTA",
    "HOND": "HONDA",
    "CHEV": "CHEVROLET",
    "CHEVY": "CHEVROLET",
    "NISS": "NISSAN",
    "HYUN": "HYUNDAI",
    "MERZ": "MERCEDES-BENZ",
    "MERCEDES": "MERCEDES-BENZ",
    "VOLK": "VOLKSWAGEN",
    "VW": "VOLKSWAGEN",
    "ACUR": "ACURA",
    "INFI": "INFINITI",
    "MITS": "MITSUBISHI",
}

MIN_VEHICLE_YEAR = 1960
MAX_VEHICLE_YEAR = 2025

# body-style shorthand that ends up in Model instead of an actual model name
# (officer probably didn't know it) - null these out
GENERIC_MODEL_CODES = {
    "4S", "2S", "4D", "4DR", "2D", "2DR", "4 DOOR", "2 DOOR",
    "SUV", "VN", "VAN", "SW", "TK", "TRUCK", "SU",
}


def load_raw_data(path=RAW_FILE):
    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        sys.exit(f"Could not find {path} - make sure it's in the project folder.")
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def normalize_column_names(df):
    # e.g. 'Personal.Injury' / 'Driver City' -> personal_injury / driver_city
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(".", "_", regex=False)
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def drop_duplicate_rows(df):
    # there are full-row dupes in the raw export - probably re-exported or double scanned records
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate rows")
    return df


def clean_boolean_columns(df):
    # leaving unrecognized values as NaN instead of False, blank isn't the same as a confirmed "no"
    yes_no_map = {
        "yes": True, "y": True,
        "no": False, "n": False,
    }
    for col in BOOLEAN_COLUMNS:
        if col not in df.columns:
            print(f"Note: expected boolean column '{col}' not found, skipping")
            continue
        mapped = df[col].astype(str).str.strip().str.lower().map(yes_no_map)
        unmapped = df[col].notna() & mapped.isna()
        if unmapped.any():
            bad_values = df.loc[unmapped, col].unique()
            print(f"'{col}': {unmapped.sum()} rows had unrecognized values {list(bad_values)[:5]}")
        df[col] = mapped
    return df


def clean_state_codes(df):
    for col in STATE_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].replace("NAN", np.nan)
        df.loc[df[col].isin(INVALID_STATE_CODES), col] = np.nan
    return df


def clean_vehicle_year(df):
    # anything outside a sane range is either a typo (2013 -> 1013) or a placeholder like 0/9999
    if "year" not in df.columns:
        return df
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    out_of_range = ~df["year"].between(MIN_VEHICLE_YEAR, MAX_VEHICLE_YEAR)
    invalid_count = (out_of_range & df["year"].notna()).sum()
    if invalid_count:
        print(f"Blanked out {invalid_count} out-of-range vehicle years")
    df.loc[out_of_range, "year"] = np.nan
    return df


def standardize_make(df):
    # collapsing TOYT/TOYOTA etc so the top-makes chart doesn't split one manufacturer into two bars
    if "make" not in df.columns:
        return df
    df["make"] = df["make"].astype(str).str.strip().str.upper()
    df["make"] = df["make"].replace(MAKE_ALIASES)
    return df


def clean_vehicle_model(df):
    if "model" not in df.columns:
        return df
    df["model"] = df["model"].astype(str).str.strip().str.upper()
    df.loc[df["model"].isin(GENERIC_MODEL_CODES), "model"] = np.nan
    return df


def categorize_violation(description):
    # order matters here - speeding check goes first so a description that
    # mentions both speeding and a plate issue still lands under Speeding
    if pd.isna(description):
        return "Unknown"

    text = str(description).lower()

    if "speed" in text or "exceeding" in text:
        return "Speeding"
    if "red light" in text or "traffic signal" in text:
        return "Red Light / Signal"
    if ("registration" in text or "expired" in text) and "plate" in text:
        return "Registration / Plate"
    if "license" in text or "suspended" in text:
        return "License / Suspended"
    if "seatbelt" in text or "belt" in text or "restrained" in text:
        return "Seatbelt"
    if "stop sign" in text:
        return "Stop Sign"
    if "alcohol" in text or "influence" in text:
        return "DUI / Alcohol"
    return "Other"


def add_accident_severity(df):
    # collapsing the 3 accident flags into one severity label, worst outcome wins
    # rows missing all three stay Unknown, not "None" - we just don't know for those
    injury = df.get("personal_injury")
    damage = df.get("property_damage")
    contributed = df.get("contributed_to_accident")

    if injury is None or damage is None or contributed is None:
        return df

    all_missing = injury.isna() & damage.isna() & contributed.isna()
    severity = np.select(
        [
            injury.fillna(False),
            damage.fillna(False),
            contributed.fillna(False),
        ],
        ["Personal Injury", "Property Damage", "Accident (No Injury/Damage Noted)"],
        default="No Accident",
    )
    severity = pd.Series(severity, index=df.index)
    severity[all_missing] = "Unknown"
    df["accident_severity"] = severity
    return df


def engineer_features(df):
    if "description" in df.columns:
        df["violation_group"] = df["description"].apply(categorize_violation)

    df = add_accident_severity(df)

    # precomputing so the dashboard isn't re-aggregating this on every rerun
    if "make" in df.columns:
        make_counts = df["make"].value_counts()
        df["make_violation_count"] = df["make"].map(make_counts)

    return df


def optimize_dtypes(df):
    # category dtype for the low-cardinality cols, saves a bit of memory on ~70k rows
    categorical_candidates = [
        "state", "driver_state", "dl_state", "race", "gender", "vehicletype",
        "violation_type", "arrest_type", "violation_group", "accident_severity",
    ]
    for col in categorical_candidates:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if "year" in df.columns:
        df["year"] = df["year"].astype("Int32")  # nullable int so NaNs survive

    return df


def clean_traffic_data():
    df = load_raw_data()
    df = normalize_column_names(df)
    df = drop_duplicate_rows(df)
    df = clean_boolean_columns(df)
    df = clean_state_codes(df)
    df = clean_vehicle_year(df)
    df = standardize_make(df)
    df = clean_vehicle_model(df)
    df = engineer_features(df)
    df = optimize_dtypes(df)

    df.to_parquet(OUTPUT_FILE, index=False, engine="pyarrow")
    print(f"\nSaved {len(df):,} cleaned rows -> {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    clean_traffic_data()
