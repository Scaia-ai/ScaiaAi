import sys
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

def summarize(src_text):
   model_name = 'google/pegasus-xsum'
   torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
   tokenizer = PegasusTokenizer.from_pretrained(model_name)
   model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
   batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest').to(torch_device)
   translated = model.generate(**batch)
   tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
   return(tgt_text)
