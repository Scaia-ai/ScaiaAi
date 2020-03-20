#!/bin/bash

wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin -O bbc/pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json -O bbc/config.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt -O bbc/vocab.txt
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json 
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

