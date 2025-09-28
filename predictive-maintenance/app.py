# app.py
import streamlit as st
import pandas as pd
from inference import run_inference


st.set_page_config(page_title="Predictive Maintenance RUL Estimation", layout="wide")

st.title("🔧 Predict Remaining Useful Life (RUL)")
st.markdown("Upload your **test dataset** (CMAPSS format) to predict RUL of engines.")

uploaded_file = st.file_uploader("Upload your test dataset (.txt or .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    with st.spinner("Running inference..."):
        # Save uploaded file temporarily
        temp_path = "temp_test.csv"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        predictions = run_inference(temp_path)

    st.success("✅ Inference completed!")

    # Show table
    st.subheader("📊 Predictions")
    st.dataframe(predictions.head(20), use_container_width=True)

    # Plot results
    st.subheader("📈 Predicted RUL per Engine")
    st.line_chart(predictions.set_index("engine_number")["pred_RUL"])

    # Download option
    csv = predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )
else:
    st.info("👆 Please upload a test file to start.")
