#!/bin/bash
mkdir -p input_data/emails
mkdir output_data
python -u get_enron.py
python process_data.py
python document_representations_bert.py
python bert_embeddings.py
python -u Ai-Advisor-enron.py
