import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gdown

# Set model directory and Google Drive file ID
model_dir = "saved_model"
model_filename = "model.safetensors"
model_path = os.path.join(model_dir, model_filename)

# 🔁 Google Drive File ID for model.safetensors
file_id = "1SItO66qXLrghG0wxOydzfbFcAgfXehrU"  # Replace with your actual file ID

# 📥 Download model from Google Drive if not present
def download_model():
    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(model_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print("📥 Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
    else:
        print("✅ model.safetensors already exists.")

# ⬇️ Ensure model is downloaded
download_model()

# 🧠 Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        from_safetensors=True
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    raise RuntimeError(
        f"❌ Error loading model from '{model_dir}'.\n"
        f"Make sure the tokenizer and model files are present.\n\nOriginal error:\n{str(e)}"
    )

# 📌 Classify clauses
def classify_clauses(df: pd.DataFrame, text_column: str = "clause") -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"❌ Column '{text_column}' not found in input DataFrame.")

    inputs = tokenizer(
        df[text_column].tolist(),
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    df["predicted_label"] = predictions
    return df
