# Exploratory analysis over cleaned_traffic.parquet. Run after clean_data.py.
# python eda.py -> eda_charts/*.png + EDA_REPORT.md
#
# No stop date/time or location in this export, so hotspot / time-of-day /
# weekday trends aren't covered - see the report's "what this data can't
# tell us" section.

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_FILE = "cleaned_traffic.parquet"
CHART_DIR = "eda_charts"
REPORT_FILE = "EDA_REPORT.md"

# same palette as app.py
BLUE = "#2a78d6"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]

sns.set_theme(style="white", rc={"axes.grid": False})


def load_cleaned_data():
    try:
        return pd.read_parquet(DATA_FILE)
    except FileNotFoundError:
        raise SystemExit(f"{DATA_FILE} not found - run clean_data.py first.")


def save_horizontal_bar(series, title, filename, color=BLUE):
    fig, ax = plt.subplots(figsize=(8, 5))
    series.sort_values().plot.barh(ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel("Number of stops")
    ax.set_ylabel("")
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, filename), dpi=150)
    plt.close(fig)


def chart_top_violations(df):
    top = df["violation_group"].value_counts().head(8)
    save_horizontal_bar(top, "Most Common Violation Categories", "top_violations.png")
    return top


def chart_vehicle_types(df):
    top = df["vehicletype"].value_counts().head(8)
    save_horizontal_bar(top, "Stops by Vehicle Type", "vehicle_types.png")
    return top


def chart_race_by_violation(df):
    top_groups = df["violation_group"].value_counts().head(5).index
    subset = df[df["violation_group"].isin(top_groups)]
    counts = pd.crosstab(subset["violation_group"], subset["race"])
    counts = counts.loc[top_groups]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    counts.plot.bar(ax=ax, color=CATEGORICAL[: len(counts.columns)])
    ax.set_title("Top Violation Categories by Driver Race")
    ax.set_ylabel("Number of stops")
    ax.set_xlabel("")
    ax.legend(title="Race", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "race_by_violation.png"), dpi=150)
    plt.close(fig)
    return counts


def chart_accident_rate_by_vehicle(df):
    rate = (
        df.groupby("vehicletype", observed=True)["contributed_to_accident"]
        .mean()
        .sort_values(ascending=False)
        .head(8)
    ) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    rate.sort_values().plot.barh(ax=ax, color="#eb6834")
    ax.set_title("Accident-Contribution Rate by Vehicle Type")
    ax.set_xlabel("% of stops where the violation contributed to an accident")
    ax.set_ylabel("")
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "accident_rate_by_vehicle.png"), dpi=150)
    plt.close(fig)
    return rate


def chart_gender_split(df):
    counts = df["gender"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(counts.index.astype(str), counts.values, color=CATEGORICAL[: len(counts)])
    ax.set_title("Stops by Gender")
    ax.set_ylabel("Number of stops")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "gender_split.png"), dpi=150)
    plt.close(fig)
    return counts


def build_report(df, top_violations, top_vehicles, accident_rate, gender_counts):
    total = len(df)
    accident_total = int(df["contributed_to_accident"].sum())
    injury_total = int(df["personal_injury"].sum())
    damage_total = int(df["property_damage"].sum())
    # group make+model together, not separately - otherwise you'd pair the most
    # common make with a model from a different make
    make_model_counts = df.dropna(subset=["make", "model"]).groupby(["make", "model"]).size()
    top_make, top_model = make_model_counts.idxmax()

    lines = [
        "# EDA Report - Traffic Violations Insight System",
        "",
        f"Cleaned dataset: **{total:,}** stops (after de-duplication).",
        "",
        "## Most Common Violations",
        "",
        "![Top violations](eda_charts/top_violations.png)",
        "",
        "Top category: **{}** ({:,} stops, {:.1f}% of all stops).".format(
            top_violations.index[0], top_violations.iloc[0], 100 * top_violations.iloc[0] / total
        ),
        "",
        "## Vehicle Types Involved",
        "",
        "![Vehicle types](eda_charts/vehicle_types.png)",
        "",
        f"Automobiles dominate the stops ({top_vehicles.iloc[0]:,}), unsurprisingly, but the "
        "accident-contribution rate tells a different story - see below.",
        "",
        "![Accident rate by vehicle](eda_charts/accident_rate_by_vehicle.png)",
        "",
        "Transit buses, school buses, and road tractors show the highest share of stops that "
        "contributed to an accident, even though they make up a small fraction of total stops. "
        "Worth flagging to a safety board even though the raw counts are small.",
        "",
        "## Demographics vs Violation Type",
        "",
        "![Race by violation](eda_charts/race_by_violation.png)",
        "",
        "![Gender split](eda_charts/gender_split.png)",
        "",
        f"Stops skew male ({gender_counts.get('M', 0):,} vs {gender_counts.get('F', 0):,} female). "
        "'Other' is consistently the largest single violation-group bucket across every race, which "
        "just reflects how varied the free-text descriptions are - see `categorize_violation()` in "
        "clean_data.py for the exact keyword rules that would need expanding to split it further.",
        "",
        "## Accidents, Injuries & Damage",
        "",
        f"- Contributed to an accident: **{accident_total:,}** stops ({100*accident_total/total:.2f}%)",
        f"- Personal injury reported: **{injury_total:,}** stops ({100*injury_total/total:.2f}%)",
        f"- Property damage reported: **{damage_total:,}** stops ({100*damage_total/total:.2f}%)",
        "",
        "## Most Frequently Cited Vehicles",
        "",
        f"Make: **{top_make}**, Model: **{top_model}**. Matches the Streamlit dashboard's "
        "vehicle-makes chart, which lets you drill into this per filter.",
        "",
        "## What This Data Can't Tell Us",
        "",
        "The spec for this project also asks about hotspots, time-of-day patterns, and "
        "weekday/month trends. `raw_traffic.csv` doesn't include a stop date, stop time, "
        "location, or lat/long - those columns exist in the full public dataset this project "
        "is modeled on, but weren't part of this export. `clean_data.py` will pick them up "
        "automatically if a richer export is dropped in later, but nothing in this report or "
        "the dashboard depends on them being there.",
    ]

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {REPORT_FILE}")


def run_eda():
    os.makedirs(CHART_DIR, exist_ok=True)
    df = load_cleaned_data()

    top_violations = chart_top_violations(df)
    top_vehicles = chart_vehicle_types(df)
    chart_race_by_violation(df)
    accident_rate = chart_accident_rate_by_vehicle(df)
    gender_counts = chart_gender_split(df)

    build_report(df, top_violations, top_vehicles, accident_rate, gender_counts)
    print(f"Charts saved to {CHART_DIR}/")


if __name__ == "__main__":
    run_eda()
