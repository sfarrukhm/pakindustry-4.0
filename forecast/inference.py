import pandas as pd
import numpy as np
import joblib
import os



def run_inference(input_file: str, output_file: str = "results/predictions.csv"):

    # ===== Load model =====
    model = joblib.load("models/best_model.pkl")

    # ===== Load test data =====
    test = pd.read_csv(input_file)


    features = [c for c in test.columns if c not in ["id", "orders", "date"]]
    for col in test.select_dtypes(include="object").columns:
        test[col] = test[col].astype("category")


    # ===== Predict =====
    preds = model.predict(test[features]).astype(int)


    # ===== Save =====
    os.makedirs("results", exist_ok=True)
    output = pd.DataFrame({"id": test.index if "id" not in test.columns else test["id"],
                           "orders": preds})
    output.to_csv(output_file, index=False)
    return output  # so Streamlit can use results


if __name__ == "__main__":
    # ===== Parse arguments =====
    import argparse
    parser = argparse.ArgumentParser(description="Run inference with trained LightGBM model")
    parser.add_argument("--input", type=str, required=True, help="Path to input test CSV (processed)")
    parser.add_argument("--output", type=str, default="results/predictions.csv", help="Path to save predictions CSV")
    args = parser.parse_args()

    # ===== Run inference =====
    predictions = run_inference(args.input,args.output)
    print(predictions.head())
    print(f"✅ Predictions saved to {args.output}")