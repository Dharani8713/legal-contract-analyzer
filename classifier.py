# classifier.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import os
import joblib

# Load trained model and tokenizer from saved directory
model_dir = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Load label encoder if you saved it (optional)
label_mapping = None

def classify_clauses(clauses):
    """
    Classifies clauses using fine-tuned Legal-BERT.

    Args:
        clauses (List[str]): List of clause texts.

    Returns:
        List[str]: Predicted clause labels.
    """
    predictions = []

    for clause in clauses:
        inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            label = label = LABELS[pred_class]
            predictions.append(label)

    return predictions

