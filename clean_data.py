"""Cleans and enriches the raw Montgomery County-style traffic violations export.

Run directly to produce cleaned_traffic.parquet from raw_traffic.csv:
    python clean_data.py

Note on scope: this particular export does not include stop date/time,
agency/location, geolocation, or search-related columns (they're part of
the full public dataset this project is modeled on, but weren't present
in raw_traffic.csv). Everything below only cleans what's actually here -
see README.md for the full list of what that means for the dashboard.
"""

import sys

import numpy as np
import pandas as pd

RAW_FILE = "raw_traffic.csv"
OUTPUT_FILE = "cleaned_traffic.parquet"

# Yes/No style columns that exist in this export. The full dataset also has
# Fatal, HAZMAT, Alcohol and Work Zone flags, but those columns never made it
# into raw_traffic.csv, so they're left out here rather than faked.
BOOLEAN_COLUMNS = [
    "belts",
    "personal_injury",
    "property_damage",
    "commercial_license",
    "commercial_vehicle",
    "contributed_to_accident",
]

STATE_COLUMNS = ["state", "driver_state", "dl_state"]

# Codes seen in the data that aren't real jurisdictions - "XX" shows up as a
# catch-all unknown and "US" is clearly a data entry mistake. Canadian
# provinces (ON, QC, MB, ...) are left alone since those are legitimate
# out-of-country drivers, not junk.
INVALID_STATE_CODES = {"XX", "US"}

# Common abbreviations/misspellings of vehicle makes found in this export.
# Not exhaustive - just the ones that show up often enough to matter for the
# "top makes" chart in the dashboard.
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

# Body-style shorthand that shows up in the Model column instead of an actual
# model name - looks like officers fell back to this when they didn't know
# the specific model. Not a real model, so it shouldn't win "most cited model".
GENERIC_MODEL_CODES = {
    "4S", "2S", "4D", "4DR", "2D", "2DR", "4 DOOR", "2 DOOR",
    "SUV", "VN", "VAN", "SW", "TK", "TRUCK", "SU",
}


def load_raw_data(path=RAW_FILE):
    """Read the raw CSV export. Bails out with a clear message if it's missing."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        sys.exit(f"Could not find {path} - make sure it's in the project folder.")
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def normalize_column_names(df):
    """lowercase_with_underscores instead of 'Personal.Injury' / 'Driver City'."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(".", "_", regex=False)
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def drop_duplicate_rows(df):
    """Full-row duplicates showed up in the raw export - most likely re-exported
    or double-scanned records rather than two separate stops that happen to
    match on every single field."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate rows")
    return df


def clean_boolean_columns(df):
    """Map the Yes/No/Y/N text columns to real booleans.

    Anything that isn't a recognized Yes/No spelling is left as missing
    (NaN) rather than silently treated as False - a blank isn't the same
    thing as a confirmed "no accident happened".
    """
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
    """Uppercase state/jurisdiction codes and null out the known junk values."""
    for col in STATE_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].replace("NAN", np.nan)
        df.loc[df[col].isin(INVALID_STATE_CODES), col] = np.nan
    return df


def clean_vehicle_year(df):
    """Vehicle model years outside a sane range (pre-1960 or newer than the
    dataset itself) are almost certainly typos - e.g. a stray digit turning
    2013 into 1013, or a placeholder like 0 / 9999 for "not entered"."""
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
    """Collapse the obvious abbreviation duplicates (TOYT/TOYOTA, HOND/HONDA, ...)
    so the "top makes" chart doesn't split one manufacturer across rows."""
    if "make" not in df.columns:
        return df
    df["make"] = df["make"].astype(str).str.strip().str.upper()
    df["make"] = df["make"].replace(MAKE_ALIASES)
    return df


def clean_vehicle_model(df):
    """Null out generic body-style codes (4S, TK, VAN, ...) so they don't
    masquerade as a real model in the "most cited model" stats - a real
    model name and a body-style shorthand shouldn't compete in the same
    ranking."""
    if "model" not in df.columns:
        return df
    df["model"] = df["model"].astype(str).str.strip().str.upper()
    df.loc[df["model"].isin(GENERIC_MODEL_CODES), "model"] = np.nan
    return df


def categorize_violation(description):
    """Bucket the free-text violation description into a handful of groups
    used throughout the dashboard. Order matters here - e.g. we check for
    "speed" before the more generic buckets so a description mentioning both
    speeding and a plate issue still lands under Speeding."""
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
    """Roll the three accident-related flags into one ordinal severity label,
    worst outcome wins. Rows missing all three flags stay Unknown instead of
    being counted as "None" - we don't actually know what happened there.
    """
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
    """Everything derived rather than read straight from the source columns."""
    if "description" in df.columns:
        df["violation_group"] = df["description"].apply(categorize_violation)

    df = add_accident_severity(df)

    # Per-make violation counts - useful for the "which vehicles show up most"
    # question without re-aggregating in the dashboard every time.
    if "make" in df.columns:
        make_counts = df["make"].value_counts()
        df["make_violation_count"] = df["make"].map(make_counts)

    return df


def optimize_dtypes(df):
    """Low-cardinality text columns are stored as category dtype instead of
    plain object strings - same values, a lot less memory on ~70k rows."""
    categorical_candidates = [
        "state", "driver_state", "dl_state", "race", "gender", "vehicletype",
        "violation_type", "arrest_type", "violation_group", "accident_severity",
    ]
    for col in categorical_candidates:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if "year" in df.columns:
        df["year"] = df["year"].astype("Int32")  # nullable int, keeps NaNs

    return df


def clean_traffic_data():
    """Run the full pipeline and write out cleaned_traffic.parquet."""
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
