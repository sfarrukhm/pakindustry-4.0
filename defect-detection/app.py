import os
import yaml
import torch
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")
# -------------------------------
# 1. CONFIGURATION
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

IMG_SIZE = config["default"]["img_size"]
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
DEFAULT_DIR = os.path.join(PROJECT_ROOT, "data", "valid")

# -------------------------------
# 2. PREPROCESSING
# -------------------------------
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(img: Image.Image, device) -> torch.Tensor:
    image = img.convert("L")
    image = transforms.ToTensor()(image).repeat(3, 1, 1)
    image = base_transform(image)
    return image.unsqueeze(0).to(device)

# -------------------------------
# 3. LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

def predict_image(img: Image.Image):
    tensor = preprocess_image(img, device)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
    label = "🟢 OK" if prob <= 0.5 else "🔴 Defected"
    return label, prob

# -------------------------------
# 4. STREAMLIT APP
# -------------------------------
st.set_page_config(page_title="Defect Detection", layout="wide")
st.title("🛠️ Cast Part Defect Detection")

# Intro section (centered & styled)
st.markdown(
    """
    <div style="text-align: center; font-size:18px;">
        Upload one or more images of cast parts.<br>
        The model will classify them as <b style="color:green;">OK</b> ✅ or 
        <b style="color:red;">Defected</b> ❌ with confidence scores.
    </div>
    """,
    unsafe_allow_html=True
)
st.write("---")

# File uploader + dataset info side by side
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_files = st.file_uploader(
        "📤 Upload cast part images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="file_uploader"
    )
with col2:
    st.caption("📂 **Default dataset directory:**")
    st.code(DEFAULT_DIR, language="bash")

# Show results in a grid layout
if uploaded_files:
    n_cols = 3  # Number of grid columns
    cols = st.columns(n_cols)

    for i, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file)
        label, prob = predict_image(img)

        # Assign to a column (grid placement)
        col = cols[i % n_cols]
        with col:
            # Card-style container
            st.markdown(
                """
                <div style="
                    border: 1px solid #ddd; 
                    border-radius: 12px; 
                    padding: 12px; 
                    margin-bottom: 20px; 
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
                ">
                """,
                unsafe_allow_html=True
            )

            st.image(img, use_container_width=True)
            st.markdown(f"**📂 File:** `{os.path.basename(uploaded_file.name)}`")

            # Prediction badge
            if "OK" in label:
                st.markdown(f"<span style='color:green; font-weight:bold;'>{label}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:red; font-weight:bold;'>{label}</span>", unsafe_allow_html=True)

            # Confidence as progress bar
            st.progress(float(prob))

            st.markdown(f"**Confidence:** {prob:.2%}")

            st.markdown("</div>", unsafe_allow_html=True)
