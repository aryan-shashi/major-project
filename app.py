import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hospital Resource Optimizer", layout="wide")

# =========================
# TITLE
# =========================
st.title("Hospital Resource Optimization System")

st.markdown("Predict patient load and optimize hospital resources")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("hospital_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df

df = load_data()

# =========================
# TRAIN MODEL
# =========================
@st.cache_resource
def train_model(df):
    X = df[["day", "month"]]
    y = df["patients"]
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

model = train_model(df)

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Input Parameters")

day = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)
month = st.sidebar.slider("Month", 1, 12, 6)

beds_available = st.sidebar.number_input("Available Beds", 50, 500, 100)
doctors_available = st.sidebar.number_input("Available Doctors", 5, 100, 10)

# =========================
# PREDICTION
# =========================
prediction = model.predict([[day, month]])[0]

beds_required = int(prediction)
doctors_required = int(prediction / 10)

# =========================
# ALERT LOGIC
# =========================
alert = None
if beds_required > beds_available:
    alert = "Bed shortage expected"
elif doctors_required > doctors_available:
    alert = "Doctor shortage expected"
else:
    alert = "Resources sufficient"

# =========================
# DISPLAY METRICS
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Predicted Patients", int(prediction))
col2.metric("Beds Required", beds_required)
col3.metric("Doctors Required", doctors_required)

# =========================
# ALERT DISPLAY
# =========================
if "shortage" in alert:
    st.error(alert)
else:
    st.success(alert)

# =========================
# HISTORICAL DATA CHART
# =========================
st.subheader("Patient Trends")

fig, ax = plt.subplots()
ax.plot(df["date"], df["patients"])
ax.set_xlabel("Date")
ax.set_ylabel("Patients")

st.pyplot(fig)

# =========================
# WHAT-IF SIMULATION
# =========================
st.subheader("What-if Scenario")

increase = st.slider("Increase in Patients (%)", 0, 100, 10)

new_prediction = int(prediction * (1 + increase / 100))

st.write(f"New Predicted Patients: {new_prediction}")

if new_prediction > beds_available:
    st.warning("⚠ Increased demand will cause bed shortage")

# =========================
# DATA VIEW
# =========================
if st.checkbox("Show Raw Data"):
    st.write(df)
