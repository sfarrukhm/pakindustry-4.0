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

IMG_SIZE = config["training"]["image_size"]
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
    image = img.convert("L")  # grayscale
    image = transforms.ToTensor()(image).repeat(3, 1, 1)  # expand to 3 channels
    image = base_transform(image)
    return image.unsqueeze(0).to(device)

# -------------------------------
# 3. LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = config["model"]["architecture"].lower()
    pretrained = config["model"]["pretrained"]

    if arch in ["efficientnet-b0", "efficientnet_b0"]:
        model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    else:
        raise ValueError(f"Model {arch} not supported in app.py")

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

st.markdown(
    """
    <div style="text-align: center; font-size:18px;">
        Upload your own images or pick from the default dataset folder.<br>
        The model will classify them as 
        <b style="color:green;">OK</b> ✅ or 
        <b style="color:red;">Defected</b> ❌ with confidence scores.
    </div>
    """,
    unsafe_allow_html=True
)
st.write("---")

# Two tabs: Upload vs Default Dataset
tab1, tab2 = st.tabs(["📤 Upload Images", "📂 Use Default Dataset"])

with tab1:
    uploaded_files = st.file_uploader(
        "Upload cast part images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="file_uploader"
    )

with tab2:
    all_valid_images = [
        os.path.join(DEFAULT_DIR, f) for f in os.listdir(DEFAULT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    selected_files = st.multiselect(
        "Select images from default dataset:",
        options=all_valid_images,
        default=all_valid_images[:5] if all_valid_images else []
    )

# Merge both sources
final_files = []
if uploaded_files:
    final_files.extend(uploaded_files)
if selected_files:
    final_files.extend(selected_files)

# Results grid
if final_files:
    n_cols = 3
    cols = st.columns(n_cols)

    for i, f in enumerate(final_files):
        if isinstance(f, str):  # From dataset (path)
            img = Image.open(f)
            fname = os.path.basename(f)
        else:  # Uploaded file
            img = Image.open(f)
            fname = os.path.basename(f.name)

        label, prob = predict_image(img)
        col = cols[i % n_cols]

        with col:
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
            st.markdown(f"**📂 File:** `{fname}`")

            if "OK" in label:
                st.markdown(f"<span style='color:green; font-weight:bold;'>{label}</span>", unsafe_allow_html=True)
                confidence = 1 - prob
            else:
                st.markdown(f"<span style='color:red; font-weight:bold;'>{label}</span>", unsafe_allow_html=True)
                confidence = prob

            st.progress(float(confidence))
            st.markdown(f"**Confidence:** {confidence:.2%}")

            st.markdown("</div>", unsafe_allow_html=True)