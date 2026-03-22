# Traffic Violations Insight System

**Exploratory Data Analysis • Data Cleaning • Interactive Streamlit Dashboard**

This project transforms raw traffic violation records into meaningful insights through data cleaning, feature engineering, and an interactive visualization dashboard built with Streamlit.

## Project Objectives

- Clean and preprocess ~70,340 traffic violation records
- Standardize column names and data types
- Map string booleans ("Yes"/"No") to actual boolean values
- Create meaningful violation categories from free-text descriptions
- Build an interactive dashboard to explore:
  - Most common violation types
  - Driver demographics (race × gender)
  - Vehicle characteristics (make, color)
  - Enforcement outcomes (charge codes, arrest types)

## Features

- Data cleaning script (`clean_data.py`)
- Interactive Streamlit dashboard (`app.py`) with:
  - Multiple filters (violation group, driver's state, arrest type)
  - Key performance metrics
  - Visualizations: bar charts, grouped bar, treemap, pie charts
  - Download filtered data as CSV
- Clean, responsive layout with expanders for better organization

## Folder Structure
traffic-insight/
├── raw_traffic.csv                 # Original raw data (input file)
├── cleaned_traffic.parquet         # Cleaned & processed data (created by clean_data.py)
├── clean_data.py                   # Data cleaning & feature engineering script
├── app.py                          # Main Streamlit dashboard application
├── README.md                       # This documentation file
└── requirements.txt                # List of required Python packages


## Requirements

```text
streamlit
pandas
plotly
pyarrow

How to Run the Project
1. Install dependencies
Open a terminal / PowerShell / Command Prompt in the project folder and run:
Bashpip install -r requirements.txt
or (on Windows if the above doesn't work):
Bashpython -m pip install streamlit pandas plotly pyarrow
2. Clean the raw data (run once)
This step processes raw_traffic.csv and creates cleaned_traffic.parquet
Bashpython clean_data.py
You should see output similar to:
textReading raw data...
Original rows: 70,340
...
Saved cleaned file → cleaned_traffic.parquet
Final rows: 70,340
3. Launch the interactive dashboard
Bashpython -m streamlit run app.py

The dashboard should automatically open in your default web browser at:
http://localhost:8501
If it doesn't open automatically, manually visit:
http://localhost:8501

4. Using the Dashboard

Sidebar filters:
Violation Group
Driver's State
Arrest Type

All metrics and charts update automatically when filters change
Use the Download Filtered Data button to export the current selection as CSV

Troubleshooting:
ProblemSolutionstreamlit command not found ->Use python -m streamlit run app.py instead
No module named 'plotly' ->Run: pip install plotly
Error loading parquet file ->Make sure you ran clean_data.py first
Dashboard shows only title / blank charts->Check terminal for errors; confirm all packages are installed
Charts not rendering->Ensure plotly and pyarrow are installed