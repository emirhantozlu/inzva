"""
Asteroid Hazard Predictor

Run with: streamlit run app.py
"""

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "asteroids.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "asteroid_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "asteroid_features.pkl"


st.set_page_config(
    page_title="Asteroid Hazard Predictor",
    page_icon="Asteroid",
    layout="wide",
)


FEATURE_LABELS = {
    "est_diameter_km": "Diameter (km)",
    "relative_velocity_km_s": "Velocity (km/s)",
    "miss_distance_mKm": "Miss Distance (million km)",
    "absolute_magnitude": "Absolute Magnitude",
    "eccentricity": "Eccentricity",
    "inclination_deg": "Inclination (deg)",
}

RISK_DIRECTIONS = {
    "est_diameter_km": "higher",
    "relative_velocity_km_s": "higher",
    "miss_distance_mKm": "lower",
    "absolute_magnitude": "lower",
    "eccentricity": "higher",
    "inclination_deg": "neutral",
}


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["hazard_label"] = df["is_hazardous"].map({1: "Hazardous", 0: "Not Hazardous"})
    return df


def build_input_frame(features, user_values):
    return pd.DataFrame([{feature: user_values[feature] for feature in features}])


def percentile_rank(series, value):
    return float((series <= value).mean() * 100)


def build_feature_importance_df(model, features):
    importance_df = pd.DataFrame(
        {
            "feature_key": features,
            "feature": [FEATURE_LABELS[feature] for feature in features],
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return importance_df


def summarize_feature_signals(df, user_values, feature_importance_df):
    hazard_df = df[df["is_hazardous"] == 1]
    non_hazard_df = df[df["is_hazardous"] == 0]
    rows = []

    for _, row in feature_importance_df.iterrows():
        feature = row["feature_key"]
        user_value = user_values[feature]
        hazard_median = float(hazard_df[feature].median())
        non_hazard_median = float(non_hazard_df[feature].median())
        direction = RISK_DIRECTIONS.get(feature, "neutral")

        risk_score = 0.0
        if direction == "higher":
            risk_score = user_value - non_hazard_median
            alignment = "risk" if user_value >= hazard_median else "mitigating"
        elif direction == "lower":
            risk_score = non_hazard_median - user_value
            alignment = "risk" if user_value <= hazard_median else "mitigating"
        else:
            alignment = "neutral"

        percentile = percentile_rank(df[feature], user_value)
        rows.append(
            {
                "feature_key": feature,
                "feature": row["feature"],
                "importance": float(row["importance"]),
                "user_value": float(user_value),
                "hazard_median": hazard_median,
                "non_hazard_median": non_hazard_median,
                "percentile": percentile,
                "alignment": alignment,
                "weighted_signal": float(risk_score * row["importance"]),
            }
        )

    signal_df = pd.DataFrame(rows)
    top_risk = signal_df[signal_df["alignment"] == "risk"].sort_values(
        "importance", ascending=False
    ).head(3)
    top_mitigating = signal_df[signal_df["alignment"] == "mitigating"].sort_values(
        "importance", ascending=False
    ).head(2)
    return signal_df, top_risk, top_mitigating


def find_similar_asteroids(df, user_values, features, n_neighbors=5):
    compare_df = df[["name", "hazard_label", "is_hazardous"] + list(features)].copy()
    std = compare_df[list(features)].std().replace(0, 1)
    deltas = (compare_df[list(features)] - pd.Series(user_values))[list(features)] / std
    compare_df["distance_score"] = np.sqrt((deltas**2).sum(axis=1))
    nearest = compare_df.nsmallest(n_neighbors, "distance_score").copy()
    nearest["distance_score"] = nearest["distance_score"].round(3)
    return nearest


def make_histogram(df, feature, user_value):
    label = FEATURE_LABELS[feature]
    hist_df = df[[feature, "hazard_label"]].copy()

    histogram = (
        alt.Chart(hist_df)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=24), title=label),
            y=alt.Y("count():Q", title="Asteroid Count"),
            color=alt.Color(
                "hazard_label:N",
                scale=alt.Scale(domain=["Not Hazardous", "Hazardous"], range=["#4c78a8", "#e45756"]),
                title="Class",
            ),
            tooltip=[
                alt.Tooltip("hazard_label:N", title="Hazard Status"),
                alt.Tooltip("count():Q", title="Count"),
            ],
        )
        .properties(height=320)
    )

    user_rule = alt.Chart(pd.DataFrame({"value": [user_value]})).mark_rule(
        color="#111827", strokeWidth=3
    ).encode(
        x="value:Q",
        tooltip=[alt.Tooltip("value:Q", title="Selected Value", format=".2f")],
    )

    return (histogram + user_rule).interactive()


def make_scatter(df, user_values):
    points = (
        alt.Chart(df)
        .mark_circle(opacity=0.55, stroke="white", strokeWidth=0.3)
        .encode(
            x=alt.X("miss_distance_mKm:Q", title="Miss Distance (million km)"),
            y=alt.Y("relative_velocity_km_s:Q", title="Velocity (km/s)"),
            size=alt.Size("est_diameter_km:Q", title="Diameter (km)", scale=alt.Scale(range=[30, 800])),
            color=alt.Color(
                "hazard_label:N",
                scale=alt.Scale(domain=["Not Hazardous", "Hazardous"], range=["#4c78a8", "#e45756"]),
                title="Hazard Status",
            ),
            tooltip=[
                alt.Tooltip("name:N", title="Asteroid"),
                alt.Tooltip("hazard_label:N", title="Hazard Status"),
                alt.Tooltip("relative_velocity_km_s:Q", title="Velocity", format=".2f"),
                alt.Tooltip("miss_distance_mKm:Q", title="Miss Distance", format=".2f"),
                alt.Tooltip("est_diameter_km:Q", title="Diameter", format=".3f"),
            ],
        )
        .properties(height=380)
    )

    selected_point = pd.DataFrame(
        [
            {
                "miss_distance_mKm": user_values["miss_distance_mKm"],
                "relative_velocity_km_s": user_values["relative_velocity_km_s"],
                "est_diameter_km": user_values["est_diameter_km"],
                "label": "Your Input",
            }
        ]
    )

    highlight = (
        alt.Chart(selected_point)
        .mark_point(shape="diamond", filled=True, size=320, color="#f2cf1d", stroke="#111827", strokeWidth=2)
        .encode(
            x="miss_distance_mKm:Q",
            y="relative_velocity_km_s:Q",
            tooltip=[
                alt.Tooltip("label:N", title="Point"),
                alt.Tooltip("relative_velocity_km_s:Q", title="Velocity", format=".2f"),
                alt.Tooltip("miss_distance_mKm:Q", title="Miss Distance", format=".2f"),
                alt.Tooltip("est_diameter_km:Q", title="Diameter", format=".3f"),
            ],
        )
    )

    return (points + highlight).interactive()


def make_dataset_mix_chart(df):
    counts = (
        df["hazard_label"]
        .value_counts()
        .rename_axis("hazard_label")
        .reset_index(name="count")
    )
    counts["share"] = counts["count"] / counts["count"].sum()

    return (
        alt.Chart(counts)
        .mark_arc(innerRadius=55, outerRadius=95)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color(
                "hazard_label:N",
                scale=alt.Scale(domain=["Not Hazardous", "Hazardous"], range=["#4c78a8", "#e45756"]),
                title="Class",
            ),
            tooltip=[
                alt.Tooltip("hazard_label:N", title="Hazard Status"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=260)
        .interactive()
    )


def make_percentile_chart(df, user_values):
    rows = []
    for feature, label in FEATURE_LABELS.items():
        value = user_values[feature]
        rows.append(
            {
                "feature": label,
                "percentile": percentile_rank(df[feature], value),
                "value": value,
            }
        )

    percentile_df = pd.DataFrame(rows)

    return (
        alt.Chart(percentile_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("percentile:Q", title="Percentile vs Dataset", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("feature:N", sort="-x", title=""),
            color=alt.Color(
                "percentile:Q",
                scale=alt.Scale(domain=[0, 50, 100], range=["#9ecae1", "#fdd0a2", "#e45756"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("value:Q", title="Selected Value", format=".3f"),
                alt.Tooltip("percentile:Q", title="Percentile", format=".1f"),
            ],
        )
        .properties(height=260)
        .interactive()
    )


def make_feature_importance_chart(feature_importance_df):
    return (
        alt.Chart(feature_importance_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("importance:Q", title="Model Importance"),
            y=alt.Y("feature:N", sort="-x", title=""),
            color=alt.value("#3b82f6"),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".3f"),
            ],
        )
        .properties(height=260)
        .interactive()
    )


model, FEATURES = load_model()
df = load_data()
feature_importance_df = build_feature_importance_df(model, FEATURES)


st.title("Asteroid Hazard Predictor")
st.caption("Interactive hazard classification demo for near-Earth asteroids")
st.markdown("---")


col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("Asteroid Parameters")
    st.write("Adjust the inputs to describe a candidate asteroid.")

    diameter = st.slider(
        "Estimated Diameter (km)",
        min_value=0.01,
        max_value=5.0,
        value=0.5,
        step=0.01,
    )
    velocity = st.slider(
        "Relative Velocity (km/s)",
        min_value=1.0,
        max_value=40.0,
        value=15.0,
        step=0.5,
    )
    miss_distance = st.slider(
        "Miss Distance (million km)",
        min_value=0.1,
        max_value=100.0,
        value=20.0,
        step=0.1,
    )
    magnitude = st.slider(
        "Absolute Magnitude",
        min_value=14.0,
        max_value=26.0,
        value=20.0,
        step=0.1,
    )
    eccentricity = st.slider(
        "Orbital Eccentricity",
        min_value=0.0,
        max_value=0.99,
        value=0.4,
        step=0.01,
    )
    inclination = st.slider(
        "Orbital Inclination (deg)",
        min_value=0.0,
        max_value=30.0,
        value=10.0,
        step=0.5,
    )

    selected_feature = st.selectbox(
        "Feature to compare against the dataset",
        options=list(FEATURES),
        format_func=lambda x: FEATURE_LABELS[x],
    )
    decision_threshold = st.slider(
        "Decision Threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        help="Lower thresholds increase recall and flag more asteroids as hazardous.",
    )


user_values = {
    "est_diameter_km": diameter,
    "relative_velocity_km_s": velocity,
    "miss_distance_mKm": miss_distance,
    "absolute_magnitude": magnitude,
    "eccentricity": eccentricity,
    "inclination_deg": inclination,
}

input_data = build_input_frame(FEATURES, user_values)
proba = float(model.predict_proba(input_data)[0][1])
prediction = int(proba >= decision_threshold)
risk_band = "High" if proba >= 0.8 else "Medium" if proba >= 0.4 else "Low"
signal_df, top_risk, top_mitigating = summarize_feature_signals(df, user_values, feature_importance_df)
nearest_asteroids = find_similar_asteroids(df, user_values, FEATURES)


with col_result:
    st.subheader("Prediction")

    if prediction == 1:
        st.error("Hazardous")
    else:
        st.success("Not Hazardous")

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Hazard Probability", f"{proba:.1%}")
    metric_2.metric("Predicted Class", "Hazardous" if prediction == 1 else "Not Hazardous")
    metric_3.metric("Risk Band", risk_band)

    risk_df = pd.DataFrame(
        [
            {"segment": "Predicted Risk", "value": proba},
            {"segment": "Remaining", "value": max(0.0, 1 - proba)},
        ]
    )

    risk_chart = (
        alt.Chart(risk_df)
        .mark_bar()
        .encode(
            x=alt.X("sum(value):Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.value(85),
            color=alt.Color(
                "segment:N",
                scale=alt.Scale(domain=["Predicted Risk", "Remaining"], range=["#e45756", "#d9e2ec"]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("segment:N"), alt.Tooltip("value:Q", format=".1%")],
        )
        .properties(height=200)
    )

    threshold_line = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(
        color="#111827", strokeDash=[6, 4]
    ).encode(x="x:Q")
    decision_line = alt.Chart(pd.DataFrame({"x": [decision_threshold]})).mark_rule(
        color="#2563eb", strokeWidth=3
    ).encode(
        x="x:Q",
        tooltip=[alt.Tooltip("x:Q", title="Decision Threshold", format=".2f")],
    )
    st.altair_chart((risk_chart + threshold_line + decision_line).interactive(), width="stretch")

    st.caption("Black dashed line = default 0.50 threshold. Blue line = current decision threshold.")

    if decision_threshold != 0.50:
        st.caption(f"Current decision threshold is {decision_threshold:.2f}.")

    with st.container(border=True):
        st.markdown("**Why this score?**")
        for _, row in top_risk.iterrows():
            st.write(
                f"- `{row['feature']}` looks risk-leaning: your value is `{row['user_value']:.2f}` "
                f"vs non-hazard median `{row['non_hazard_median']:.2f}`."
            )
        for _, row in top_mitigating.iterrows():
            st.write(
                f"- `{row['feature']}` looks safer than typical hazardous cases: your value is "
                f"`{row['user_value']:.2f}` vs hazard median `{row['hazard_median']:.2f}`."
            )


st.markdown("---")
st.subheader("Interactive Comparison Views")

compare_col_1, compare_col_2 = st.columns([1.2, 1])

with compare_col_1:
    st.write("Velocity vs miss distance across the dataset. The gold diamond marks your input.")
    st.altair_chart(make_scatter(df, user_values), width="stretch")

with compare_col_2:
    st.write("Distribution by class for the selected feature. Hover for counts and class details.")
    st.altair_chart(
        make_histogram(df, selected_feature, user_values[selected_feature]),
        width="stretch",
    )


st.markdown("---")
st.subheader("Quick Context for Non-Technical Users")

simple_col_1, simple_col_2 = st.columns(2)

with simple_col_1:
    st.write("How common are hazardous asteroids in this sample?")
    st.altair_chart(make_dataset_mix_chart(df), width="stretch")

with simple_col_2:
    st.write("Where does your asteroid sit versus the dataset?")
    st.altair_chart(make_percentile_chart(df, user_values), width="stretch")

st.info(
    "Reading the visuals: larger diameter, higher velocity, and lower miss distance generally push risk upward. "
    "A high percentile bar means your selected value is larger than most asteroids in the dataset."
)


st.markdown("---")
st.subheader("Model Perspective")

model_col_1, model_col_2 = st.columns([1, 1.1])

with model_col_1:
    st.write("Which variables matter most to the model?")
    st.altair_chart(make_feature_importance_chart(feature_importance_df), width="stretch")

with model_col_2:
    st.write("Closest historical examples in the dataset")
    neighbor_view = nearest_asteroids[
        [
            "name",
            "hazard_label",
            "distance_score",
            "relative_velocity_km_s",
            "miss_distance_mKm",
            "est_diameter_km",
        ]
    ].rename(
        columns={
            "name": "Asteroid",
            "hazard_label": "Hazard Status",
            "distance_score": "Similarity Distance",
            "relative_velocity_km_s": "Velocity (km/s)",
            "miss_distance_mKm": "Miss Distance (million km)",
            "est_diameter_km": "Diameter (km)",
        }
    )
    st.dataframe(neighbor_view, width="stretch", hide_index=True)


st.markdown("---")
st.subheader("Dataset Explorer")

filter_col_1, filter_col_2, filter_col_3 = st.columns(3)
with filter_col_1:
    show_haz = st.selectbox(
        "Filter by class",
        ["All", "Hazardous only", "Not Hazardous only"],
    )
with filter_col_2:
    sort_by = st.selectbox("Sort by", FEATURES, format_func=lambda x: FEATURE_LABELS[x])
with filter_col_3:
    n_rows = st.slider("Rows to show", 5, 50, 10)

display_df = df.copy()
if show_haz == "Hazardous only":
    display_df = display_df[display_df["is_hazardous"] == 1]
elif show_haz == "Not Hazardous only":
    display_df = display_df[display_df["is_hazardous"] == 0]

display_df = display_df.sort_values(sort_by, ascending=False).head(n_rows).copy()
display_df["is_hazardous"] = display_df["is_hazardous"].map({1: "Yes", 0: "No"})
st.dataframe(display_df, width="stretch")


st.markdown("---")
stat_1, stat_2, stat_3, stat_4 = st.columns(4)
stat_1.metric("Total Asteroids", len(df))
stat_2.metric("Hazardous", int(df["is_hazardous"].sum()))
stat_3.metric("Not Hazardous", int((df["is_hazardous"] == 0).sum()))
stat_4.metric("Hazard Rate", f"{df['is_hazardous'].mean():.1%}")

st.caption("Built from the saved workshop model and asteroid feature dataset.")
