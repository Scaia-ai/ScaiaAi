import pandas as pd

#df = pd.read_csv('../test-data/collected_02/metadata_0.csv', sep='|')
df = pd.read_csv('/home/mark/projects/scaia/scaia-test-data/collected_02/metadata_0_first_lines.csv', sep='|')


def tagger(src_text):
    if not src_text or src_text == "" or not isinstance(src_text, str):
        return ("")
    # tgt_text = src_text.split()
    # tgt_text = src_text.split()
    tgt_text = "word1"
    return(tgt_text)

#print(summarize([df.loc[0, 'text']]))


df['tag1'] = df['text'].apply(tagger)

df[['text', 'tag1']].to_csv('output.csv', sep='|')
