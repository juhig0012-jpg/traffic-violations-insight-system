import streamlit as st
import pandas as pd
import plotly.express as px

# ──────────────────────────────────────────────────────────────
# Page config & styling
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Violations Insight Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance (optional but nice)
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    .block-container {
        padding-top: 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Load data with caching & error handling
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet("cleaned_traffic.parquet")
        # Convert booleans properly
        bool_cols = ['belts', 'personal_injury', 'property_damage',
                     'commercial_license', 'commercial_vehicle', 'contributed_to_accident']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        return df
    except Exception as e:
        st.error(f"Could not load cleaned_traffic.parquet\nError: {str(e)}")
        st.error("Run clean_data.py first or check file path.")
        st.stop()

with st.spinner("Loading data..."):
    df = load_data()

# ──────────────────────────────────────────────────────────────
# Header & description
# ──────────────────────────────────────────────────────────────
st.title("Traffic Violations Insight Dashboard")
st.markdown("""
**Interactive analysis of traffic violations**  
Showing **{:,}** processed records • Educational project  
Data source: raw_traffic.csv (cleaned & enriched)
""".format(len(df)))

# ──────────────────────────────────────────────────────────────
# Sidebar – Filters
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    # Violation Group
    if 'violation_group' in df.columns:
        groups = sorted(df['violation_group'].unique())
        selected_groups = st.multiselect(
            "Violation Group",
            options=groups,
            default=["Speeding", "Registration / Plate", "License / Suspended"]
        )

    # State (Driver's state)
    if 'state' in df.columns:
        states = sorted(df['state'].dropna().unique())
        selected_states = st.multiselect("Driver's State", states, default=["MD"])

    # Arrest Type
    if 'arrest_type' in df.columns:
        arrest_types = sorted(df['arrest_type'].dropna().unique())
        selected_arrest = st.multiselect("Arrest Type", arrest_types)

    # Reset filters button
    if st.button("Reset all filters"):
        st.rerun()

# ──────────────────────────────────────────────────────────────
# Apply filters
# ──────────────────────────────────────────────────────────────
filtered = df.copy()

if 'violation_group' in filtered.columns and selected_groups:
    filtered = filtered[filtered['violation_group'].isin(selected_groups)]

if 'state' in filtered.columns and selected_states:
    filtered = filtered[filtered['state'].isin(selected_states)]

if 'arrest_type' in filtered.columns and selected_arrest:
    filtered = filtered[filtered['arrest_type'].isin(selected_arrest)]

if len(filtered) == 0:
    st.warning("No records match the current filters. Try clearing some selections.")
    st.stop()

# ──────────────────────────────────────────────────────────────
# Key metrics row
# ──────────────────────────────────────────────────────────────
st.subheader("Summary Statistics")
cols = st.columns(4)
cols[0].metric("Total Violations", f"{len(filtered):,}", delta=None)
cols[1].metric("Accident Contributed", f"{filtered['contributed_to_accident'].sum():,}")
cols[2].metric("Seatbelt Violations", f"{filtered['belts'].sum():,}")
cols[3].metric("Personal Injury Cases", f"{filtered['personal_injury'].sum():,}")

# Download button
csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Data (CSV)",
    data=csv,
    file_name=f"filtered_traffic_violations_{len(filtered)}rows.csv",
    mime="text/csv",
)

# ──────────────────────────────────────────────────────────────
# Charts – organized in expanders
# ──────────────────────────────────────────────────────────────
with st.expander("Violation Categories & Patterns", expanded=True):
    col_left, col_right = st.columns(2)

    with col_left:
        # Top Violation Groups
        st.subheader("Top Violation Groups")
        top_viol = filtered['violation_group'].value_counts().head(12).reset_index()
        top_viol.columns = ['Group', 'Count']
        fig_viol = px.bar(
            top_viol,
            x='Count',
            y='Group',
            orientation='h',
            color='Count',
            color_continuous_scale='Viridis',
            title="Most Common Violation Categories"
        )
        st.plotly_chart(fig_viol, use_container_width=True)

    with col_right:
        # Top Charge Codes
        st.subheader("Top 10 Charge Codes")
        if 'charge' in filtered.columns:
            top_charge = filtered['charge'].value_counts().head(10).reset_index()
            top_charge.columns = ['Charge', 'Count']
            fig_charge = px.bar(
                top_charge,
                x='Count',
                y='Charge',
                orientation='h',
                title="Most Frequent Charge Codes",
                color='Count',
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig_charge, use_container_width=True)

with st.expander("Demographics", expanded=False):
    # Race & Gender
    st.subheader("Violations by Race and Gender")
    if 'race' in filtered.columns and 'gender' in filtered.columns:
        rg = filtered.groupby(['race', 'gender']).size().reset_index(name='Count')
        fig_rg = px.bar(
            rg,
            x='race',
            y='Count',
            color='gender',
            barmode='group',
            title="Race × Gender Distribution",
            height=500
        )
        st.plotly_chart(fig_rg, use_container_width=True)

with st.expander("Vehicle Information", expanded=False):
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        # Top Vehicle Makes – Treemap
        st.subheader("Top Vehicle Makes")
        if 'make' in filtered.columns:
            makes = filtered['make'].value_counts().head(15).reset_index()
            makes.columns = ['Make', 'Count']
            fig_make = px.treemap(
                makes,
                path=['Make'],
                values='Count',
                color='Count',
                color_continuous_scale='Blues',
                title="Top Vehicle Makes Involved in Violations"
            )
            st.plotly_chart(fig_make, use_container_width=True)

    with col_v2:
        # Top Vehicle Colors – Pie
        st.subheader("Top Vehicle Colors")
        if 'color' in filtered.columns:
            colors = filtered['color'].value_counts().head(10)
            fig_color = px.pie(
                colors,
                values=colors.values,
                names=colors.index,
                title="Most Common Vehicle Colors",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_color, use_container_width=True)

with st.expander("Enforcement & Arrest", expanded=False):
    # Arrest Type
    st.subheader("Arrest Type Distribution")
    if 'arrest_type' in filtered.columns:
        arrest_counts = filtered['arrest_type'].value_counts()
        fig_arrest = px.pie(
            arrest_counts,
            values=arrest_counts.values,
            names=arrest_counts.index,
            title="Citation vs Warning vs Other",
            hole=0.3
        )
        st.plotly_chart(fig_arrest, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Data processed from raw_traffic.csv • Educational project • Built with Streamlit & Plotly")