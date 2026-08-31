"""Streamlit dashboard for the cleaned traffic violations dataset.

Run with:
    streamlit run app.py

Expects cleaned_traffic.parquet to already exist (produced by clean_data.py).
"""

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_FILE = "cleaned_traffic.parquet"

# Same categorical order and blue sequential ramp used in eda.py, so a chart
# doesn't mean something different depending on whether you're looking at the
# static report or the live dashboard.
CATEGORICAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
BLUE_SCALE = [[0, "#cde2fb"], [1, "#0d366b"]]

FILTER_KEYS = ["violation_group", "state", "arrest_type", "gender", "race", "vehicletype"]


def configure_page():
    st.set_page_config(
        page_title="Traffic Violations Insight Dashboard",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stMetric { background-color: #f0f2f6; border-radius: 10px; padding: 10px; }
        .block-container { padding-top: 1rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data():
    """Load the cleaned parquet file. Raises so the caller can show a
    Streamlit-friendly error instead of a bare traceback."""
    df = pd.read_parquet(DATA_FILE)
    bool_cols = [
        "belts", "personal_injury", "property_damage",
        "commercial_license", "commercial_vehicle", "contributed_to_accident",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def render_sidebar_filters(df):
    """Draws every filter widget and returns the selections as a dict of
    column -> list of chosen values (empty list means "no filter applied")."""
    with st.sidebar:
        st.header("Filters")

        selections = {}

        if "violation_group" in df.columns:
            groups = sorted(df["violation_group"].dropna().unique())
            selections["violation_group"] = st.multiselect(
                "Violation Group", groups,
                default=["Speeding", "Registration / Plate", "License / Suspended"],
                key="filter_violation_group",
            )

        if "state" in df.columns:
            states = sorted(df["state"].dropna().unique())
            selections["state"] = st.multiselect(
                "Driver's State", states, default=["MD"], key="filter_state"
            )

        if "vehicletype" in df.columns:
            vehicle_types = sorted(df["vehicletype"].dropna().unique())
            selections["vehicletype"] = st.multiselect(
                "Vehicle Type", vehicle_types, key="filter_vehicletype"
            )

        if "gender" in df.columns:
            genders = sorted(df["gender"].dropna().unique())
            selections["gender"] = st.multiselect(
                "Gender", genders, key="filter_gender"
            )

        if "race" in df.columns:
            races = sorted(df["race"].dropna().unique())
            selections["race"] = st.multiselect(
                "Race", races, key="filter_race"
            )

        if "arrest_type" in df.columns:
            arrest_types = sorted(df["arrest_type"].dropna().unique())
            selections["arrest_type"] = st.multiselect(
                "Arrest Type", arrest_types, key="filter_arrest_type"
            )

        st.button("Reset all filters", on_click=reset_filters)

    return selections


def reset_filters():
    """Callback for the reset button - clears every filter widget's stored
    state before Streamlit re-runs, so the multiselects actually go back to
    empty instead of just triggering a no-op rerun."""
    for key in FILTER_KEYS:
        widget_key = f"filter_{key}"
        if widget_key in st.session_state:
            st.session_state[widget_key] = []


def apply_filters(df, selections):
    filtered = df
    for column, chosen in selections.items():
        if chosen:
            filtered = filtered[filtered[column].isin(chosen)]
    return filtered


def render_summary_metrics(filtered):
    st.subheader("Summary Statistics")
    cols = st.columns(4)
    cols[0].metric("Total Violations", f"{len(filtered):,}")
    cols[1].metric("Accident Contributed", f"{filtered['contributed_to_accident'].sum():,}")
    cols[2].metric("Seatbelt Violations", f"{filtered['belts'].sum():,}")
    cols[3].metric("Personal Injury Cases", f"{filtered['personal_injury'].sum():,}")

    if {"make", "model"}.issubset(filtered.columns) and len(filtered):
        combo = filtered.dropna(subset=["make", "model"]).groupby(["make", "model"]).size()
        if len(combo):
            top_make, top_model = combo.idxmax()
            st.caption(
                f"Most frequently cited vehicle in the current selection: "
                f"**{top_make} {top_model}** ({combo.max():,} stops)"
            )


def render_violation_charts(filtered):
    with st.expander("Violation Categories & Patterns", expanded=True):
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Top Violation Groups")
            top_viol = filtered["violation_group"].value_counts().head(12).reset_index()
            top_viol.columns = ["Group", "Count"]
            fig = px.bar(
                top_viol, x="Count", y="Group", orientation="h",
                color="Count", color_continuous_scale=BLUE_SCALE,
                title="Most Common Violation Categories",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Top 10 Charge Codes")
            if "charge" in filtered.columns:
                top_charge = filtered["charge"].value_counts().head(10).reset_index()
                top_charge.columns = ["Charge", "Count"]
                fig = px.bar(
                    top_charge, x="Count", y="Charge", orientation="h",
                    color="Count", color_continuous_scale=BLUE_SCALE,
                    title="Most Frequent Charge Codes",
                )
                st.plotly_chart(fig, use_container_width=True)


def render_demographics_charts(filtered):
    with st.expander("Demographics", expanded=False):
        st.subheader("Violations by Race and Gender")
        if {"race", "gender"}.issubset(filtered.columns):
            rg = filtered.groupby(["race", "gender"]).size().reset_index(name="Count")
            fig = px.bar(
                rg, x="race", y="Count", color="gender", barmode="group",
                color_discrete_sequence=CATEGORICAL_COLORS,
                title="Race × Gender Distribution", height=500,
            )
            st.plotly_chart(fig, use_container_width=True)


def render_vehicle_charts(filtered):
    with st.expander("Vehicle Information", expanded=False):
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Top Vehicle Makes")
            if "make" in filtered.columns:
                makes = filtered["make"].value_counts().head(15).reset_index()
                makes.columns = ["Make", "Count"]
                fig = px.treemap(
                    makes, path=["Make"], values="Count",
                    color="Count", color_continuous_scale=BLUE_SCALE,
                    title="Top Vehicle Makes Involved in Violations",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Vehicle Type Breakdown")
            if "vehicletype" in filtered.columns:
                types = filtered["vehicletype"].value_counts().head(10).reset_index()
                types.columns = ["Vehicle Type", "Count"]
                fig = px.bar(
                    types, x="Count", y="Vehicle Type", orientation="h",
                    color_discrete_sequence=[CATEGORICAL_COLORS[0]],
                    title="Stops by Vehicle Type",
                )
                st.plotly_chart(fig, use_container_width=True)


def render_enforcement_charts(filtered):
    with st.expander("Enforcement & Arrest", expanded=False):
        st.subheader("Arrest Type Distribution")
        if "arrest_type" in filtered.columns:
            arrest_counts = filtered["arrest_type"].value_counts().reset_index()
            arrest_counts.columns = ["Arrest Type", "Count"]
            fig = px.bar(
                arrest_counts, x="Count", y="Arrest Type", orientation="h",
                color_discrete_sequence=[CATEGORICAL_COLORS[1]],
                title="Citations, Warnings & Other Outcomes",
            )
            st.plotly_chart(fig, use_container_width=True)


def render_download_button(filtered):
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Data (CSV)",
        data=csv_bytes,
        file_name=f"filtered_traffic_violations_{len(filtered)}rows.csv",
        mime="text/csv",
    )


def main():
    configure_page()

    with st.spinner("Loading data..."):
        try:
            df = load_data()
        except FileNotFoundError:
            st.error(f"Could not find {DATA_FILE}. Run `python clean_data.py` first.")
            st.stop()

    st.title("Traffic Violations Insight Dashboard")
    st.markdown(
        f"**Interactive analysis of traffic violations** \n"
        f"Showing **{len(df):,}** processed records • Educational project \n"
        f"Data source: raw_traffic.csv (cleaned & enriched)"
    )

    selections = render_sidebar_filters(df)
    filtered = apply_filters(df, selections)

    if filtered.empty:
        st.warning("No records match the current filters. Try clearing some selections.")
        st.stop()

    render_summary_metrics(filtered)
    render_download_button(filtered)
    render_violation_charts(filtered)
    render_demographics_charts(filtered)
    render_vehicle_charts(filtered)
    render_enforcement_charts(filtered)

    st.markdown("---")
    st.caption("Data processed from raw_traffic.csv • Educational project • Built with Streamlit & Plotly")


if __name__ == "__main__":
    main()
