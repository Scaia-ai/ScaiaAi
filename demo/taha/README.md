# Taha Demo

## Environment

You need to create a conda environment called `huggingface`

```bash
conda create -n huggingface tensorflow pytorch torchvision cudatoolkit=10.1 -c pytorch
```


## Run Script 

run the following: `download_huggingface.sh`
This only should be done once


## Run the main:

run the following: `main.sh`


## About this

This contains the following:
`main.py`
creates a model based on SQUAD 2.0 using `run_squad.py` from huggingface library.
fine-tunes it with extra sample file "dev-test.json". tuned model wll be saved in out2 directory

test.py
gets a context and a question, then finds the answer using learnt model out2.

and also SQUAD 2.0 and other data-sets.

notes:
1-BertForQuestionAnswer needs one of the BERT pre-trained models, main.py assumes that "bert-base-case"  is downloaded to the bbc directory, but if you've ever used it, it is probably in your cache and you can use it by setting `--model_name_or_path` to "bert-base-case" , else you have two options:
a) download it to the directory bbc
b) set `--model_name_or_path` to `bert-base-case  and let it to be downloaded its about 400Mb

2- My system is too slow, so I set `--max_steps` to 2 just to check the code and get a model, so the result is not good, if you want to have a good results unset this limitation and if you have a powerful GPU, I suggest these setting both for first model and fine tune process:

```bash
--num_train_epochs 10 \
--max_seq_length 384 \
--doc_stride 128 \
--per_gpu_eval_batch_size=32   \
--per_gpu_train_batch_size=32   \
```
