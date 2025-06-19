# ner.py

import spacy

# Load SpaCy's English NER model
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
