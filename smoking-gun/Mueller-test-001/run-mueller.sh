#!/bin/bash
python -u Split_PDF_Paragraphs.py
mkdir output_data
python process_data.py
python document_representations_bert.py
python bert_embeddings.py
python -u Ai-Advisor-Mueller-Test.py
