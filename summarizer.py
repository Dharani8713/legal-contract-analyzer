# summarizer.py
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Load once
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def summarize_contract(text, max_length=200, min_length=60):
    """
    Summarizes contract text using BART.
    Args:
        text (str): Input legal text.
        max_length (int): Max tokens in summary.
        min_length (int): Min tokens in summary.
    Returns:
        str: Summary text.
    """
    inputs = tokenizer.batch_encode_plus([text], return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
