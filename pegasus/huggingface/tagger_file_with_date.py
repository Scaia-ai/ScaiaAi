import os

# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# import torch

import pandas as pd
from collections import Counter

file_path = '/home/mark/projects/scaia/scaia-test-data/jeff/metadata_0.csv'
file_name = os.path.basename(file_path)

df = pd.read_csv(file_path, sep='|')

if (os.path.exists("output") == False):
    os.mkdir("output")

def dater(src_text):
    if not src_text or src_text == "" or not isinstance(src_text, str):
        return ("")
    my_date = "2020-11-01"
    return(my_date)

df['file_date'] = df['text'].apply(dater)
df['tag'] = "whatever"

# Output final results
df[['tag', 'file_date']].to_csv('output/' + file_name + '_tags.csv', sep='|', index=False)