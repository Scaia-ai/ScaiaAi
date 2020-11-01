import os

# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# import torch

import pandas as pd
from collections import Counter

#df = pd.read_csv('../test-data/collected_02/metadata_0.csv', sep='|')
file_path = '/home/mark/projects/scaia/scaia-test-data/jeff/metadata_0.csv'
file_name = os.path.basename(file_path)

df = pd.read_csv(file_path, sep='|')

if (os.path.exists("output") == False):
    os.mkdir("output")

#model_name = 'google/pegasus-xsum'
#torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
#tokenizer = PegasusTokenizer.from_pretrained(model_name)
#model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)


# def summarize(src_text):
#    if not src_text or src_text == "" or not isinstance(src_text, str):
#        return ("")
#    batch = tokenizer.prepare_seq2seq_batch([src_text], truncation=True, padding='longest').to(torch_device)
#    translated = model.generate(**batch)
#    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
#    return(tgt_text)


def tagger(src_text, tag_number):
    if not src_text or src_text == "" or not isinstance(src_text, str):
        return ("")
    tag_number_i = int(tag_number)
    split_it = src_text.split()
    count_it = Counter(split_it)
    most_common = count_it.most_common(10)
    tgt_text = most_common[tag_number_i]
    return(tgt_text[0])

# Summarize
# df['summary'] = df['text'].apply(summarize)
# Output intermediary results
# df[['summary', 'text']].to_csv('output/' + file_name + '_summary.csv', sep='|')
# df[['summary']].to_csv('output/' + file_name + '_summary.csv', sep='|')
# Apply tag
df['tag0'] = df['text'].apply(tagger, tag_number = "0")
df['tag1'] = df['text'].apply(tagger, tag_number = "1")
df['tag2'] = df['text'].apply(tagger, tag_number = "2")
df['tag3'] = df['text'].apply(tagger, tag_number = "3")
df['tag4'] = df['text'].apply(tagger, tag_number = "4")
# Output final results
df[['tag0', 'tag1', 'tag2', 'tag3', 'tag4']].to_csv('output/' + file_name + '_tags.csv', sep='|', index=False)