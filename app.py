import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Road Safety Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_DIR = Path(__file__).parent
ACCIDENT_YEAR = 2024


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    return {
        "caract": pd.read_csv(DATA_DIR / "caract-2024.csv", sep=";"),
        "lieux": pd.read_csv(DATA_DIR / "lieux-2024.csv", sep=";"),
        "usagers": pd.read_csv(DATA_DIR / "usagers-2024.csv", sep=";"),
        "vehicules": pd.read_csv(DATA_DIR / "vehicules-2024.csv", sep=";"),
    }


SEMANTIC_MAPS = {
    "caract": {
        "Num_Acc": "Accident identifier",
        "jour": "Day of month",
        "mois": "Month",
        "an": "Year",
        "hrmn": "Time of accident",
        "lum": "Lighting conditions",
        "dep": "Department code",
        "com": "Commune code",
        "agg": "Urban area indicator",
        "int": "Intersection type",
        "atm": "Weather conditions",
        "col": "Collision type",
        "adr": "Address / location text",
        "lat": "Latitude of the accident location",
        "long": "Longitude of the accident location",
    },
    "lieux": {
        "Num_Acc": "Accident identifier",
        "catr": "Road category",
        "voie": "Road name / route",
        "v1": "Road number / local index",
        "v2": "Road suffix / extension",
        "circ": "Traffic regime",
        "nbv": "Number of lanes",
        "vosp": "Reserved lane / cycle lane presence",
        "prof": "Longitudinal profile",
        "pr": "PR marker",
        "pr1": "PR sub-marker",
        "plan": "Horizontal alignment",
        "lartpc": "Central reservation / median width",
        "larrout": "Roadway width",
        "surf": "Surface condition",
        "infra": "Infrastructure",
        "situ": "Road situation",
        "vma": "Maximum authorized speed",
    },
    "usagers": {
        "Num_Acc": "Accident identifier",
        "id_usager": "User identifier",
        "id_vehicule": "Vehicle identifier",
        "num_veh": "Vehicle number in the accident",
        "place": "Occupant position",
        "catu": "User category",
        "grav": "Injury severity",
        "sexe": "Sex",
        "an_nais": "Birth year",
        "trajet": "Journey purpose",
        "secu1": "Primary safety equipment",
        "secu2": "Secondary safety equipment",
        "secu3": "Tertiary safety equipment",
        "locp": "Pedestrian location",
        "actp": "Pedestrian action",
        "etatp": "Pedestrian condition",
    },
    "vehicules": {
        "Num_Acc": "Accident identifier",
        "id_vehicule": "Vehicle identifier",
        "num_veh": "Vehicle number in the accident",
        "senc": "Direction of travel",
        "catv": "Vehicle category",
        "obs": "Stationary obstacle",
        "obsm": "Moving obstacle",
        "choc": "Point of impact",
        "manv": "Maneuver",
        "motor": "Motorization / engine type",
        "occutc": "Occupancy field for collective transport",
    },
}


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #334155 100%);
        color: white;
        padding: 1.5rem 1.75rem;
        border-radius: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18);
    }
    .hero h1, .hero p { margin: 0; }
    .hero p { opacity: 0.88; margin-top: 0.45rem; }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        padding: 1rem 1rem 0.75rem 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    .section-title { margin-top: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


data = load_data()
caract = data["caract"]
lieux = data["lieux"]
usagers = data["usagers"]
vehicules = data["vehicules"]


# Bronze metrics
all_tables = {"caract": caract, "lieux": lieux, "usagers": usagers, "vehicules": vehicules}
row_counts = {name: len(df) for name, df in all_tables.items()}
duplicate_counts = {name: int(df.duplicated().sum()) for name, df in all_tables.items()}
missing_percent = {name: round(float(df.isna().mean().mean() * 100), 2) for name, df in all_tables.items()}
missing_columns = {
    name: int((df.isna().mean() * 100 >= 20).sum())
    for name, df in all_tables.items()
}

coords = caract[["lat", "long"]].copy()
coords["lat_num"] = pd.to_numeric(coords["lat"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
coords["long_num"] = pd.to_numeric(coords["long"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
invalid_lat = int(((coords["lat_num"] < -90) | (coords["lat_num"] > 90)).sum())
invalid_long = int(((coords["long_num"] < -180) | (coords["long_num"] > 180)).sum())
invalid_coords = invalid_lat + invalid_long

age = ACCIDENT_YEAR - pd.to_numeric(usagers["an_nais"], errors="coerce")
invalid_age = int(((age < 0) | (age > 110)).sum())
missing_birth_year = int(usagers["an_nais"].isna().sum())

negative_code_cols = []
negative_code_count = 0
for name, df in all_tables.items():
    for column in df.select_dtypes(include="number").columns:
        count = int((df[column] < 0).sum())
        if count > 0:
            negative_code_cols.append((name, column, count))
            negative_code_count += count

life_tables = {
    "Bronze": "Raw CSV extracts loaded without modification.",
    "Silver": "Standardized, cleaned, deduplicated, and feature-enriched tables.",
    "Gold": "Curated analytical tables and model-ready outputs.",
}


st.markdown(
    "<div class='hero'><h1>Road Safety Data Dashboard</h1><p>From raw CSV files to a cleaned Silver layer, then to KPI-ready Gold outputs.</p></div>",
    unsafe_allow_html=True,
)

st.sidebar.header("Dashboard controls")
selected_table = st.sidebar.selectbox("Inspect table", list(all_tables.keys()), index=0)
show_tail = st.sidebar.checkbox("Show cleaned sample rows", value=True)

st.subheader("Executive KPIs")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accident rows", f"{row_counts['caract']:,}")
col2.metric("User rows", f"{row_counts['usagers']:,}")
col3.metric("Vehicle rows", f"{row_counts['vehicules']:,}")
col4.metric("Lieux duplicates", f"{duplicate_counts['lieux']}")
col5.metric("Invalid coords", f"{invalid_coords}")
col6.metric("Missing birth year", f"{missing_birth_year:,}")

st.markdown("---")
left, right = st.columns([1.05, 0.95])

with left:
    st.subheader("Main data quality signals")
    quality_df = pd.DataFrame(
        [
            {"table": name, "rows": row_counts[name], "duplicates": duplicate_counts[name], "avg_missing_%": missing_percent[name], "columns_missing_20%+": missing_columns[name]}
            for name in all_tables
        ]
    )
    fig = px.bar(
        quality_df,
        x="table",
        y=["duplicates", "columns_missing_20%+"],
        barmode="group",
        title="Duplicates and sparse columns by table",
        color_discrete_sequence=["#0f766e", "#ca8a04"],
    )
    fig.update_layout(height=420, legend_title_text="Metric")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Missingness overview")
    missing_rows = []
    for name, df in all_tables.items():
        for column, pct in (df.isna().mean() * 100).sort_values(ascending=False).items():
            if pct > 0:
                missing_rows.append({"table": name, "column": column, "missing_%": round(float(pct), 2)})
    missing_table = pd.DataFrame(missing_rows).sort_values("missing_%", ascending=False).head(12)
    st.dataframe(missing_table, use_container_width=True, hide_index=True)

st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Problem areas")
    problem_df = pd.DataFrame(
        [
            {"issue": "High missingness in lieux.lartpc", "impact": "Almost unusable without special handling"},
            {"issue": "High missingness in vehicules.occutc", "impact": "Low analytical value as-is"},
            {"issue": "Duplicate rows in lieux", "impact": "Can overweight some road situations"},
            {"issue": "Negative sentinel codes", "impact": "Need recoding before modeling"},
            {"issue": "Birth year missingness", "impact": "Affects age-based features"},
        ]
    )
    st.dataframe(problem_df, use_container_width=True, hide_index=True)

with col_b:
    st.subheader("Recommended Silver transformations")
    st.markdown(
        """
        - **Standardize** dates, times, coordinates, and category codes.
        - **Clean** invalid or sentinel values by recoding them to missing.
        - **Deduplicate** exact repeated records in `lieux`.
        - **Enrich** with `age`, `hour`, and `time_of_day`.
        - **Document** every step so the pipeline remains auditable.
        """
    )

st.markdown("---")
st.subheader("Selected table explorer")
selected_df = all_tables[selected_table]
selected_missing = (selected_df.isna().mean() * 100).round(2).reset_index()
selected_missing.columns = ["column", "missing_%"]
selected_missing["dtype"] = selected_df.dtypes.astype(str).values
selected_missing["meaning"] = selected_missing["column"].map(SEMANTIC_MAPS[selected_table]).fillna("")

exp1, exp2 = st.columns(2)
with exp1:
    st.dataframe(selected_missing.sort_values("missing_%", ascending=False), use_container_width=True, hide_index=True)
with exp2:
    if show_tail:
        st.dataframe(selected_df.head(10), use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Silver features and baseline model")
merged = caract[["Num_Acc", "hrmn", "lum", "atm", "col"]].merge(
    usagers[["Num_Acc", "grav", "sexe", "an_nais", "catu", "trajet"]], on="Num_Acc", how="inner"
)
merged = merged.replace(-1, np.nan)
merged["age"] = ACCIDENT_YEAR - pd.to_numeric(merged["an_nais"], errors="coerce")
merged["hour"] = pd.to_numeric(merged["hrmn"].astype(str).str.extract(r"^(\d{1,2})")[0], errors="coerce")
merged["time_of_day"] = pd.cut(
    merged["hour"],
    bins=[-1, 5, 11, 17, 20, 24],
    labels=["night", "morning", "afternoon", "evening", "late_night"],
)

feature_summary = pd.DataFrame(
    {
        "feature": ["age", "sexe", "catu", "trajet", "lum", "atm", "hour", "time_of_day"],
        "role": ["numeric", "categorical", "categorical", "categorical", "categorical", "categorical", "numeric", "categorical"],
        "why it helps": [
            "Captures driver age effects",
            "Basic demographic signal",
            "User type",
            "Journey purpose",
            "Lighting conditions",
            "Weather conditions",
            "Time-based severity pattern",
            "Time-of-day context",
        ],
    }
)

model_kpi1, model_kpi2, model_kpi3 = st.columns(3)
model_kpi1.metric("Model rows", f"{len(merged):,}")
model_kpi2.metric("Engineered features", "8")
model_kpi3.metric("Target classes", f"{merged['grav'].nunique()}")

st.dataframe(feature_summary, use_container_width=True, hide_index=True)

st.markdown(
    """
    **Baseline model proposal:** multinomial logistic regression to predict `grav` from the cleaned Silver features.
    It is easy to explain, quick to train, and gives a transparent benchmark before moving to more powerful models.
    """
)

st.markdown("---")
st.subheader("Medallion architecture")
arch1, arch2, arch3 = st.columns(3)
with arch1:
    st.info("**Bronze**\n\nRaw CSV files: original extracts from data.gouv.fr.\nNo cleaning, no recoding.")
with arch2:
    st.success("**Silver**\n\nStandardized, cleaned, deduplicated, and enriched tables.\nReady for analytics and modeling.")
with arch3:
    st.warning("**Gold**\n\nCurated KPIs, dashboards, and model-ready outputs.\nOptimized for decision-making.")

st.graphviz_chart(
    """
    digraph G {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fontname="Helvetica"];
        bronze [label="Bronze\nRaw CSV files", fillcolor="#dbeafe"];
        silver [label="Silver\nCleaned + standardized + deduplicated", fillcolor="#dcfce7"];
        gold [label="Gold\nKPIs, dashboards, model-ready features", fillcolor="#fef3c7"];
        features [label="Feature engineering\nage, time_of_day, cleaned categories", fillcolor="#e0e7ff"];
        model [label="Baseline model\nLogistic regression on severity", fillcolor="#fee2e2"];
        decision [label="Decision support", fillcolor="#f8fafc"];

        bronze -> silver;
        silver -> gold;
        silver -> features;
        features -> model;
        gold -> decision;
        model -> decision;
    }
    """
)

st.markdown("---")
st.subheader("Documentation summary")
st.markdown(
    """
    - Negative codes such as `-1` should be recoded to nulls before the Silver layer.
    - Exact duplicates in `lieux` should be removed.
    - `lartpc` and `occutc` are too sparse for most general analyses.
    - `adr` and `an_nais` can be retained but flagged or imputed depending on the business question.
    - Coordinates and age values are usable after standardization.
    """
)
