import os

import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

if (os.path.exists("output") == False):
    os.mkdir("output")

#df = pd.read_csv('../test-data/collected_02/metadata_0.csv', sep='|')
df = pd.read_csv('/home/mark/projects/scaia/scaia-test-data/collected_02/metadata_0_first_lines.csv', sep='|')

model_name = 'google/pegasus-xsum'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)


def summarize(src_text):
    if not src_text or src_text == "" or not isinstance(src_text, str):
        return ("")
    batch = tokenizer.prepare_seq2seq_batch([src_text], truncation=True, padding='longest').to(torch_device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return(tgt_text)

#print(summarize([df.loc[0, 'text']]))

df['summary'] = df['text'].apply(summarize)

df[['summary', 'text']].to_csv('output/summary.csv', sep='|')
