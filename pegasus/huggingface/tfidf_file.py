import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# import torch

import pandas as pd
from collections import Counter

#df = pd.read_csv('../test-data/collected_02/metadata_0.csv', sep='|')
# file_path = '/home/tfox/scaia/scaia-test-data/jeff/metadata_0.csv'
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



def tfidf(src_text, num_terms):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 5, stop_words='english')
    tfidf_matrix =  tf.fit_transform(src_text)
    feature_names = np.array(tf.get_feature_names())
    tfidf_sorting = src_text.apply(lambda x: feature_names[np.argsort(tf.transform([x]).toarray()).flatten()[::-1]][:num_terms])
    return tfidf_sorting





# Summarize
# df['summary'] = df['text'].apply(summarize)
# Output intermediary results
# df[['summary', 'text']].to_csv('output/' + file_name + '_summary.csv', sep='|')
# df[['summary']].to_csv('output/' + file_name + '_summary.csv', sep='|')
# Apply tag

df.text = df.text.fillna('')

df['tags'] = tfidf(df['text'], 5)
df['tag0'] = df['tags'].apply(lambda x : x[0])
df['tag1'] = df['tags'].apply(lambda x : x[1])
df['tag2'] = df['tags'].apply(lambda x : x[2])
df['tag3'] = df['tags'].apply(lambda x : x[3])
df['tag4'] = df['tags'].apply(lambda x : x[4])
# Output final results
df[['tag0', 'tag1', 'tag2', 'tag3', 'tag4']].to_csv('output/' + file_name + '_tags.csv', sep='|', index=False)
