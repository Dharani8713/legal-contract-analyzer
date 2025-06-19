import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np

# Load model and tokenizer
model_dir = "nlpaueb/legal-bert-base-uncased"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
except Exception as e:
    raise RuntimeError(
        f"❌ Error loading model from '{model_dir}'.\nMake sure you have trained and saved it.\nOriginal error:\n{e}"
    )

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Classify clauses
def classify_clauses(clauses: list[str]) -> list[str]:
    inputs = tokenizer(clauses, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.cpu().numpy().tolist()


# Optional: Training script when run directly
if __name__ == "__main__":
    print("🔧 Training model from scratch...")

    # Load your dataset (make sure this file exists and has 'clause' + 'label' columns)
    df = pd.read_csv("data/clauses.csv")  # Example path

    # Encode labels
    label_map = {label: i for i, label in enumerate(sorted(df['label'].unique()))}
    df['label'] = df['label'].map(label_map)

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)

    def tokenize(batch):
        return tokenizer(batch["clause"], padding=True, truncation=True)

    tokenized = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir="./saved_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=1,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
    )

    trainer.train()
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")
    print("✅ Model trained and saved to ./saved_model")
