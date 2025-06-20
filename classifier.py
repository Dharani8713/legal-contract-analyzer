from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_dir = "./saved_model"

# Load from safetensors
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir,
    use_safetensors=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify_clauses(clauses):
    inputs = tokenizer(clauses, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.cpu().numpy()
