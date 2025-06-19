# app.py

import streamlit as st
import pandas as pd
import tempfile
import os

from document_parser import (
    extract_text_from_pdf,
    extract_text_from_docx,
    clean_text,
    chunk_text_into_paragraphs
)
from ner import extract_entities
from classifier import classify_clauses
from summarizer import summarize_contract

st.set_page_config(page_title="Legal Contract Analyzer", layout="wide")
st.title("📑 AI-Powered Legal Contract Analyzer")

uploaded_file = st.file_uploader("Upload a contract (.pdf or .docx)", type=["pdf", "docx"])

if uploaded_file:
    suffix = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Step 1: Extract and clean text
    if suffix == "pdf":
        raw_text = extract_text_from_pdf(tmp_path)
    elif suffix == "docx":
        raw_text = extract_text_from_docx(tmp_path)
    else:
        st.error("❌ Unsupported file format.")
        st.stop()

    cleaned_text = clean_text(raw_text)
    clauses = chunk_text_into_paragraphs(cleaned_text)

    if not clauses:
        st.error("❌ No usable clauses found after parsing. Try another file.")
        st.stop()

    # Step 2: Named Entity Recognition
    with st.spinner("🔎 Extracting entities..."):
        entities = extract_entities(cleaned_text)
        entity_df = pd.DataFrame(entities)

    # Step 3: Clause Classification
    with st.spinner("⚖️ Classifying clauses..."):
        clause_labels = classify_clauses(clauses)
        clause_df = pd.DataFrame({
            "Clause": clauses,
            "Label": clause_labels
        })

    # Step 4: Summarize full document
    with st.spinner("📝 Generating summary..."):
        summary = summarize_contract(cleaned_text)

    # Display outputs
    st.subheader("📜 Clauses and Classification")
    st.dataframe(clause_df)

    st.subheader("🏷️ Extracted Named Entities")
    st.dataframe(entity_df)

    st.subheader("🧠 Summary of Contract")
    st.write(summary)

    # Prepare and download results
    json_data = {
        "clauses": clauses,
        "labels": clause_labels,
        "entities": entities,
        "summary": summary
    }

    clause_csv = clause_df.to_csv(index=False)
    clause_json = pd.DataFrame(json_data).to_json(orient="records")

    st.download_button("📥 Download Clauses (CSV)", clause_csv, file_name="clauses.csv")
    st.download_button("📥 Download Results (JSON)", clause_json, file_name="contract_analysis.json")

    st.success("✅ Contract analysis complete!")
