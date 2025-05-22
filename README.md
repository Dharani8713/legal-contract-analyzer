# 🧠 AI-Powered Legal Contract Analyzer

This project uses NLP techniques to extract clauses from legal contracts (e.g., NDAs) using the CUAD dataset.

## 🚀 Setup Instructions

1. Clone the repo:
git clone https://github.com/Dharani8713/legal-contract-analyzer.git


2. Create a conda env:
conda create -n contractenv python=3.10
conda activate contractenv


3. Install dependencies:
pip install -r requirements.txt


4. Download `CUAD_v1.json` from:
[CUAD GitHub](https://github.com/TheAtticusProject/cuad)

Place it in: `data/raw/`

## 📊 Dataset
We use the CUAD dataset (Contract Understanding Atticus Dataset) for clause extraction tasks.

## 📁 Folder Structure

legal-contract-analyzer/
├── data/
│ └── raw/
│ └── CUAD_v1.json (not included)
├── notebooks/
│ └── explore_cuad.ipynb
├── app.py (or streamlit_app.py)
└── README.md

