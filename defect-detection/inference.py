import os
import torch
import yaml
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# -------------------------------
# 1. CONFIGURATION
# -------------------------------
with open("./src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

IMG_SIZE = config["default"]["img_size"]
MODEL_PATH = os.path.join("models", "best_model.pth")

# -------------------------------
# 2. PREPROCESSING PIPELINE
# -------------------------------
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# 3. LOAD MODEL
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=None)  # only structure
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print(f"✅ Model loaded from {MODEL_PATH} on {device}")

# -------------------------------
# 4. INFERENCE UTILITIES
# -------------------------------
def preprocess_image(img: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for model inference.
    Converts grayscale → RGB, resizes, normalizes, returns batch tensor.
    """
    image = img.convert("L")                            # grayscale
    image = transforms.ToTensor()(image).repeat(3, 1, 1)  # expand to 3 channels
    image = base_transform(image)                       # resize + normalize
    return image.unsqueeze(0).to(device)                # add batch dim

def predict_image(img_input) -> tuple:
    """
    Run inference on a single image.
    Args:
        img_input (str | PIL.Image.Image): path or PIL image
    Returns:
        (label: str, probability: float)
    """
    if isinstance(img_input, str):       # if path provided
        img = Image.open(img_input)
    elif isinstance(img_input, Image.Image):  # if PIL image provided
        img = img_input
    else:
        raise TypeError("img_input must be a file path or PIL.Image")

    tensor = preprocess_image(img)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    label = "Defected" if prob > 0.5 else "OK"
    return label, prob

def predict_folder(folder_path: str):
    """Run inference on all images in a folder and print summary."""
    ok_count, defect_count = 0, 0
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(folder_path, fname)
        try:
            label, prob = predict_image(img_path)
            print(f"🖼️ {fname} → {label} ({prob:.2%})")
            if label == "OK":
                ok_count += 1
            else:
                defect_count += 1
        except Exception as e:
            print(f"⚠️ Skipped {fname} ({e})")

    print("\n📊 Summary:")
    print(f"  OK: {ok_count}")
    print(f"  Defected: {defect_count}")

# -------------------------------
# 5. RUN DEMO
# -------------------------------
if __name__ == "__main__":
    # Single image test
    test_image = "defect-detection/data/valid/cast_def_0_15_jpeg.rf.25a9b096e676969ad5ff4fe1e5ee8153.jpg"
    if os.path.exists(test_image):
        label, prob = predict_image(test_image)
        print(f"Single test → {os.path.basename(test_image)}: {label} ({prob:.2%})")

    # Folder test
    test_folder = "defect-detection/data/valid"
    if os.path.exists(test_folder):
        predict_folder(test_folder)
