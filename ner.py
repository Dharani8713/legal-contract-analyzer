# ner.py

import spacy

# ✅ Load SpaCy's small English NER model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extract named entities from the input contract text using SpaCy.

    Args:
        text (str): Contract content as a string.

    Returns:
        List[Dict]: A list of extracted entities with text, label, and position.
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
