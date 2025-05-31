import spacy

def load_model():
    return spacy.load("ner_model")

def predict(text):
    nlp = load_model()
    doc = nlp(text)
    for ent in doc.ents:
        print(f"{ent.text} --> {ent.label_}")

if __name__ == "__main__":
    sample_text = """The Receiving Party shall maintain all information in strict confidentiality.
    This agreement shall terminate in 2 years."""
    predict(sample_text)
