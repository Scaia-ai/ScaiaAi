#!/bin/bash
mkdir -p input_data/results/text
mkdir output_data
python -u get_freed.py
python process_data.py
python document_representations_bert.py
python bert_embeddings.py
python -u Ai-Advisor-freed.py