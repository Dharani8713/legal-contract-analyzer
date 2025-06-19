import os
import json
import csv
import pdfplumber
import docx2txt
import re

def extract_text_from_pdf(file_path):
    text = ''
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def clean_text(text):
    # Remove excessive whitespace and normalize
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
    text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with a single space
    return text.strip()

def chunk_text_into_paragraphs(text):
    # Split by line breaks to simulate clauses or paragraphs
    return [para.strip() for para in text.split('\n') if para.strip()]

def save_as_json(chunks, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=4)

def save_as_csv(chunks, output_file):
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Clause_ID', 'Text'])
        for i, chunk in enumerate(chunks):
            writer.writerow([f'C{i+1}', chunk])

def process_document(file_path, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        raw_text = extract_text_from_pdf(file_path)
    elif ext == 'docx':
        raw_text = extract_text_from_docx(file_path)
    else:
        print(f"Unsupported file type: {ext}")
        return

    cleaned_text = clean_text(raw_text)
    chunks = chunk_text_into_paragraphs(cleaned_text)

    ext = os.path.splitext(file_path)[1].replace('.', '')
    base_name = os.path.splitext(os.path.basename(file_path))[0] + f"_{ext}"

    json_output = os.path.join(output_dir, f'{base_name}_clauses.json')
    csv_output = os.path.join(output_dir, f'{base_name}_clauses.csv')

    save_as_json(chunks, json_output)
    save_as_csv(chunks, csv_output)

    print(f"[✓] Processed: {file_path}")
    print(f"    → Saved JSON to: {json_output}")
    print(f"    → Saved CSV to : {csv_output}")

# Example usage:
if __name__ == "__main__":
    # Put your sample files inside the 'samples' folder
    sample_files = [
        'samples/sample_contract.pdf',
        'samples/sample_contract.docx'
    ]

    for file_path in sample_files:
        if os.path.exists(file_path):
            process_document(file_path)
        else:
            print(f"[!] File not found: {file_path}")
