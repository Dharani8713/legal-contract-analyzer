# ner.py

import spacy
import subprocess
import importlib.util

# ✅ Ensure the SpaCy model is available, otherwise download it
def ensure_spacy_model():
    model_name = "en_core_web_sm"
    if importlib.util.find_spec(model_name) is None:
        print("🔄 Downloading en_core_web_sm...")
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)

# ⬇️ Run the check-and-download logic
ensure_spacy_model()

# 🧠 Load the SpaCy model
nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    """
    Extracts named entities from contract text using SpaCy.

    Parameters:
        text (str): The full contract text.

    Returns:
        List[Dict]: A list of dictionaries with entity text, label, and start/end position.
    """
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        entities.append({
            "Text": ent.text,
            "Label": ent.label_,
            "Start_Char": ent.start_char,
            "End_Char": ent.end_char
        })

    return entities
