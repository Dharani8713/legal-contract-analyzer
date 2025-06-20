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


## Installation

## Setup Guide
1. Clone the repo:
   git clone <your_repo_url>

2. Install dependencies:
   pip install -r requirements.txt

3. (Optional) Download model:
   python download_model.py  # if model.safetensors is not in repo

4. Launch the app:
   streamlit run app.py
