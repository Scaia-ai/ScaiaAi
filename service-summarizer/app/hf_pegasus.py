import sys
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch

def summarize(src_text):
   model_name = 'google/pegasus-xsum'
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   tokenizer = PegasusTokenizer.from_pretrained(model_name)
   model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
   batch = tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)
   translated = model.generate(**batch)
   tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
   return(tgt_text)

def summarize_legal(src_text):
   tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")
   model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus")
   input_tokenized = tokenizer.encode(src_text, return_tensors='pt',max_length=1024,truncation=True)
   summary_ids = model.generate(input_tokenized,
                                  num_beams=9,
                                  no_repeat_ngram_size=3,
                                  length_penalty=2.0,
                                  min_length=150,
                                  max_length=250,
                                  early_stopping=True)
   summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
   return(summary)


