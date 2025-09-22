import streamlit as st
from PIL import Image
from inference import predict_image

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Cast Defect Detection", page_icon="🛠️")

# -------------------------------
# 1. HEADER
# -------------------------------
st.title("🛠️ Cast Part Defect Detection")
st.write("Upload a cast part image to check if it’s **OK** or **Defected**.")

# -------------------------------
# 2. FILE UPLOADER
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run prediction
    with st.spinner("🔎 Analyzing..."):
        label, prob = predict_image(image)

    # Display result
    st.markdown(f"### ✅ Prediction: **{label}**")
    st.markdown(f"**Confidence:** {prob:.2%}")

    # Color-coded result box
    if label == "Defected":
        st.error("⚠️ Defect Detected in Part")
    else:
        st.success("✅ Part is OK")
