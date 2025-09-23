import os
import time
import argparse
import yaml
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

# -------------------------------
# 1. CONFIGURATION
# -------------------------------
with open("./src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

IMG_SIZE = config["training"]["image_size"]
MODEL_ARCH = config["model"]["architecture"]
MODEL_PATH = os.path.join("models", "best_model.pth")
DEVICE = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")

# Preprocessing (consistent with training)
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# 2. LOAD MODEL
# -------------------------------
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()
print(f"✅ Model loaded from {MODEL_PATH} on {DEVICE}")

# -------------------------------
# 3. UTILITIES
# -------------------------------
def preprocess_image(img: Image.Image) -> torch.Tensor:
    image = img.convert("L")                            # grayscale
    image = transforms.ToTensor()(image).repeat(3, 1, 1)  # expand to 3 channels
    image = base_transform(image)
    return image.unsqueeze(0).to(DEVICE)                # add batch dim

def predict_image(img_path, return_confidence=False):
    img = Image.open(img_path)
    tensor = preprocess_image(img)

    start_time = time.time()
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
    elapsed = time.time() - start_time

    label = "Defected" if prob > 0.5 else "OK"

    result = {
        "image": img_path,
        "prediction": label,
        "confidence": prob,
        "time": elapsed
    }

    if return_confidence:
        print(f"Image: {img_path}")
        print(f"Prediction: {label}")
        print(f"Confidence: {prob:.2f} ({prob:.1%})")
        print(f"Processing time: {elapsed:.2f} seconds\n")
    else:
        print(f"{os.path.basename(img_path)} → {label} ({prob:.1%})")

    return result

def predict_folder(folder_path, output_csv=None, visualize=False):
    results = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(folder_path, fname)
        try:
            result = predict_image(img_path)
            results.append(result)
        except Exception as e:
            print(f"⚠️ Skipped {fname} ({e})")

    df = pd.DataFrame(results)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"✅ Results saved to {output_csv}")

    if visualize:
        for r in results[:10]:  # show up to 10 images
            img = Image.open(r["image"])
            plt.imshow(img)
            plt.title(f"{r['prediction']} ({r['confidence']:.1%})")
            plt.axis("off")
            plt.show()

# -------------------------------
# 4. CLI ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Defect Detection Inference")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument("--output", type=str, help="CSV file to save results (for folder mode)")
    parser.add_argument("--confidence", action="store_true", help="Show confidence score for single image")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions (folder mode)")
    args = parser.parse_args()

    if args.image:
        predict_image(args.image, return_confidence=args.confidence)
    elif args.folder:
        predict_folder(args.folder, output_csv=args.output, visualize=args.visualize)
    else:
        print("⚠️ Please provide either --image or --folder")
