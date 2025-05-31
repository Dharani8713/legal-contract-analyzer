# AI-Powered Legal Contract Analyzer

A Python project that parses legal contract documents (PDF and DOCX), extracts clauses and named entities using Natural Language Processing (NLP) techniques, specifically SpaCy-based Named Entity Recognition (NER).  
This project aims to help automate contract review by identifying key legal clauses and entities.

---

## Features

- Parse legal PDFs and DOCX files using `pdfplumber` and `docx2txt`
- Clean and chunk contract documents into clauses or paragraphs
- Train and fine-tune SpaCy NER models to identify clause types (e.g., Confidentiality, Termination)
- Generate outputs in JSON and CSV formats for further analysis

---

## Project Structure

legal-contract-analyzer/
├── document_parser.py # Script to parse and chunk documents
├── ner_model.py # Script for training/inference of NER model
├── train_ner.ipynb # Jupyter notebook for NER model training
├── train_data_spacy.json # Sample training data for NER
├── requirements.txt # Python dependencies
├── README.md # Project overview
├── samples/ # Sample contract files (PDF and DOCX)
│ ├── sample_contract.pdf
│ └── sample_contract.docx
└── outputs/ # Parsed outputs (JSON, CSV)
├── sample_contract_clauses.json
└── sample_contract_clauses.csv

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Dharani8713/legal-contract-analyzer.git
   cd legal-contract-analyzer
2. Create a Python virtual environment and activate it:

On Windows:
python -m venv venv
venv\Scripts\activate
On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

3. Install required packages:
   pip install -r requirements.txt

Usage
Parse documents:
python document_parser.py --input samples/sample_contract.pdf --output_dir outputs/
This command will parse the input contract file, chunk it into clauses, and save outputs in JSON and CSV formats inside the outputs/ folder.

Train NER model
Open and run the Jupyter notebook train_ner.ipynb to train or fine-tune the SpaCy NER model on your annotated data.

Evaluate NER model
Run the evaluation script (to be added) to measure model performance on a test set and generate an evaluation report.

Dataset
The training data format follows SpaCy's JSON format for NER and includes clause annotations inspired by the CUAD dataset (Contract Understanding Atticus Dataset).
