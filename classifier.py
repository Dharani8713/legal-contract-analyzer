# classifier.py
import os
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# Load model and tokenizer
model_dir = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Define device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your label list
LABELS = [
    "Label 0", "Label 1", "Label 2", "Label 3", "Label 4", "Label 5",
    "Label 6", "Label 7", "Label 8", "Label 9", "Label 10", "Label 11",
    "Label 12", "Label 13", "Label 14", "Label 15", "Label 16", "Label 17",
    "Label 18", "Label 19", "Label 20", "Label 21", "Label 22", "Label 23",
    "Label 24", "Label 25", "Label 26", "Label 27", "Label 28", "Label 29",
    "Label 30", "Label 31", "Label 32", "Label 33", "Label 34", "Label 35",
    "Label 36", "Label 37", "Label 38"
]
def classify_clauses(clauses):
    predictions = []
    for clause in clauses:
        inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

            # DEBUG PRINT
            print("Predicted class index:", pred_class)
            print("Model output logits shape:", outputs.logits.shape)
            print("Number of labels available:", len(LABELS))

            if pred_class >= len(LABELS):
             print(f"[!] Warning: pred_class {pred_class} out of LABELS range")
             label = "Unknown"
            else:
              label = LABELS[pred_class]
            predictions.append(label)

    return predictions

