import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gdown

# 📁 Path to model directory
model_dir = "saved_model"
model_file = os.path.join(model_dir, "model.safetensors")

# 🔗 Google Drive direct download URL (replace with your real file ID)
gdrive_file_id = "1SItO66qXLrghG0wxOydzfbFcAgfXehrU"
gdrive_url = f"https://drive.google.com/uc?id={gdrive_file_id}"

# ⬇ Download model.safetensors if missing
if not os.path.exists(model_file):
    os.makedirs(model_dir, exist_ok=True)
    print("⏬ Downloading model.safetensors from Google Drive...")
    gdown.download(gdrive_url, model_file, quiet=False)
else:
    print("✅ model.safetensors already exists.")

# 🧠 Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    raise RuntimeError(
        f"❌ Error loading model from '{model_dir}'.\n"
        f"Make sure the tokenizer and model files are present.\n\nOriginal error:\n{e}"
    )

# 📊 Clause classification function
def classify_clauses(paragraphs):
    model.eval()
    predictions = []

    with torch.no_grad():
        for paragraph in paragraphs:
            inputs = tokenizer(paragraph, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
            predictions.append(predicted_class_id)

    return predictions
