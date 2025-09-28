import streamlit as st
import pandas as pd
from inference import run_inference

st.set_page_config(page_title="Supply Chain Forecasting", layout="wide")
st.title("📦 Supply Chain Demand Forecasting Dashboard")

st.markdown("Upload a processed test CSV (with lag features) to generate forecasts.")

uploaded_file = st.file_uploader(
    "Upload a test CSV (e.g., sample_test.csv from ./data/forecast/)",
    type=["csv"]
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = f"data/forecast/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run inference
    output = run_inference(temp_path)

    # Display
    st.subheader("📊 Predictions")
    st.dataframe(output.head(20))

    st.subheader("📈 Summary Statistics")
    st.write(output["orders"].describe())

    st.subheader("🏭 Forecast Distribution by Warehouse")
    output["warehouse"] = output["id"].str.split("_").str[0]
    st.bar_chart(output.groupby("warehouse")["orders"].mean())

    # Download
    csv = output.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")
else:
    st.info("ℹ️ Please upload a CSV from ./data/forecast/ to continue.")
