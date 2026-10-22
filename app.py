import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

st.title("Road Accident Severity Classification")
st.write("Educational demo using the supplied dataset. It is not a real-world safety tool.")

@st.cache_resource
def train_demo():
    df = pd.read_excel("indian_roads_dataset.xlsx")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df = df.drop(columns=[c for c in ["accident_id","state","date","time","cause","vehicles_involved","casualties","festival","risk_score"] if c in df.columns])
    X = df.drop(columns=["accident_severity"])
    y = df["accident_severity"]
    ord_cols = [c for c in ["visibility","traffic_density"] if c in X.columns]
    cat_cols = [c for c in ["city","road_type","weather","day_of_week"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in ord_cols + cat_cols]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("enc",OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ("ord", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("enc",OrdinalEncoder(categories=[["low","medium","high"]]*len(ord_cols),handle_unknown="use_encoded_value",unknown_value=-1))]), ord_cols)
    ])
    model = Pipeline([("pre",pre),("clf",DecisionTreeClassifier(class_weight="balanced",random_state=42))])
    model.fit(X,y)
    return model, X.columns.tolist()

model, columns = train_demo()
st.subheader("Enter sample conditions")
city = st.selectbox("City", ["Bangalore","Chandigarh","Chennai","Delhi","Hyderabad","Kolkata","Mumbai","Pune"])
road_type = st.selectbox("Road type", ["urban","rural","highway"])
weather = st.selectbox("Weather", ["clear","rain","fog"])
visibility = st.selectbox("Visibility", ["low","medium","high"])
traffic_density = st.selectbox("Traffic density", ["low","medium","high"])
day_of_week = st.selectbox("Day of week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
hour = st.slider("Hour",0,23,12)
latitude = st.number_input("Latitude", value=19.0)
longitude = st.number_input("Longitude", value=73.0)
lanes = st.slider("Number of lanes",1,6,3)
temperature = st.slider("Temperature",15,40,28)
traffic_signal = st.selectbox("Traffic signal present?", [0,1])
is_weekend = st.selectbox("Weekend?", [0,1])
is_peak_hour = st.selectbox("Peak hour?", [0,1])
year = st.slider("Year",2022,2025,2024)
month = st.slider("Month",1,12,6)
row = {
    "city":city,"latitude":latitude,"longitude":longitude,"hour":hour,
    "day_of_week":day_of_week,"is_weekend":is_weekend,"road_type":road_type,
    "lanes":lanes,"traffic_signal":traffic_signal,"weather":weather,
    "visibility":visibility,"temperature":temperature,"traffic_density":traffic_density,
    "is_peak_hour":is_peak_hour,"year":year,"month":month,
    "hour_sin":np.sin(2*np.pi*hour/24),"hour_cos":np.cos(2*np.pi*hour/24)
}
sample = pd.DataFrame([row])
sample = sample.reindex(columns=columns)
if st.button("Predict severity"):
    prediction = model.predict(sample)[0]
    st.success(f"Predicted class: {prediction}")
    st.caption("Educational demonstration only. The dataset may be synthetic and model performance on real incidents is not established.")
