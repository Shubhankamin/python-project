
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Road Accident Severity",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Road Accident Severity Classification")
st.write(
    "Enter the road and weather conditions to get an experimental "
    "accident-severity classification."
)

st.warning(
    "Educational demo only. Predictions are not suitable for real-world "
    "road-safety or emergency decisions."
)


@st.cache_resource
def train_demo():
    df = pd.read_excel("indian_roads_dataset.xlsx")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    drop_cols = [
        "accident_id", "state", "date", "time", "cause",
        "vehicles_involved", "casualties", "festival", "risk_score"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X = df.drop(columns=["accident_severity"])
    y = df["accident_severity"]

    ord_cols = ["visibility", "traffic_density"]
    cat_cols = ["city", "road_type", "weather", "day_of_week"]
    num_cols = [
        c for c in X.columns
        if c not in ord_cols + cat_cols
    ]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),

        ("ord", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=[["low", "medium", "high"]] * len(ord_cols),
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ]), ord_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X, y)
    return model, X.columns.tolist()


model, columns = train_demo()

st.subheader("Road and environmental conditions")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox(
        "City",
        ["Bangalore", "Chandigarh", "Chennai", "Delhi",
         "Hyderabad", "Kolkata", "Mumbai", "Pune"]
    )

    road_type = st.selectbox(
        "Road type", ["urban", "rural", "highway"]
    )

    weather = st.selectbox(
        "Weather", ["clear", "rain", "fog"]
    )

    visibility = st.selectbox(
        "Visibility", ["low", "medium", "high"]
    )

with col2:
    traffic_density = st.selectbox(
        "Traffic density", ["low", "medium", "high"]
    )

    traffic_signal = st.selectbox(
        "Traffic signal present?", [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    lanes = st.selectbox("Number of lanes", [1, 2, 3, 4, 5, 6])

    temperature = st.slider("Temperature (°C)", 15, 40, 28)

hour = st.slider("Hour of day", 0, 23, 12)
day_of_week = st.selectbox(
    "Day of week",
    ["Monday", "Tuesday", "Wednesday", "Thursday",
     "Friday", "Saturday", "Sunday"]
)

is_weekend = int(day_of_week in ["Saturday", "Sunday"])
is_peak_hour = int(hour in [7, 8, 9, 17, 18, 19])

# Keep location and date-related model features internally.
# These are representative defaults, not user-provided measurements.
row = {
    "city": city,
    "latitude": 19.0,
    "longitude": 73.0,
    "hour": hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "road_type": road_type,
    "lanes": lanes,
    "traffic_signal": traffic_signal,
    "weather": weather,
    "visibility": visibility,
    "temperature": temperature,
    "traffic_density": traffic_density,
    "is_peak_hour": is_peak_hour,
    "year": 2024,
    "month": 6,
    "hour_sin": np.sin(2 * np.pi * hour / 24),
    "hour_cos": np.cos(2 * np.pi * hour / 24)
}

sample = pd.DataFrame([row]).reindex(columns=columns)

if st.button("Predict severity", type="primary", use_container_width=True):
    prediction = model.predict(sample)[0]

    st.subheader("Prediction")
    st.success(f"Predicted severity: {str(prediction).title()}")

    st.caption(
        "This is an experimental prediction from the supplied dataset. "
        "It is not a reliable estimate of real-world accident risk."
    )
