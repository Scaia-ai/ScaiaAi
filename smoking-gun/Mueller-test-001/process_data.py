# ####################################################################
# to process documents (i.e, emails & save them into csv)
# ####################################################################

import os, re
import pandas as pd


def process_pages(data_folder = os.path.join("input_data", "pages"), page_text_file = os.path.join("input_data", "page_texts.csv"),
                  text_filter=""):

    df = pd.DataFrame(columns=["filename", "text"])
    df.to_csv(page_text_file, index=False)


    for ix, fn in enumerate(os.listdir(data_folder)):

        with open(os.path.join(data_folder, fn), "r", encoding="utf-8") as fp:
            text = fp.readlines()

        # cleaning text of newlines and strings with only special characters
        text = " ".join(filter(lambda x: re.sub('[^A-Za-z0-9]+', '', x.replace(text_filter, "")), text))

        if text.strip()=="":
            continue
        
        tmp_df = pd.DataFrame(columns=["filename", "text"])
        tmp_df.loc[0] = [fn, text]
        with open(page_text_file, 'a', encoding="utf-8") as pf:
            tmp_df.to_csv(pf, header=False, index=False)

    return

def process_paragraphs(data_folder = os.path.join("input_data", "paragraphs"), paragraph_text_file = os.path.join("input_data", "paragraph_texts.csv"),
                  text_filter=""):

    df = pd.DataFrame(columns=["filename", "text"])
    df.to_csv(paragraph_text_file, index=False)


    for ix, fn in enumerate(os.listdir(data_folder)):
        
        if fn== '.DS_Store': 
            continue

        with open(os.path.join(data_folder, fn), "r", encoding="utf-8") as fp:
            text = fp.readlines()

        # cleaning text of newlines and strings with only special characters
        text = " ".join(filter(lambda x: re.sub('[^A-Za-z0-9]+', '', x.replace(text_filter, "")), text))

        if text.strip()=="":
            continue
        
        tmp_df = pd.DataFrame(columns=["filename", "text"])
        tmp_df.loc[0] = [fn, text]
        with open(paragraph_text_file, 'a', encoding="utf-8") as pf:
            tmp_df.to_csv(pf, header=False, index=False)

    return

if __name__ == "__main__":
    process_paragraphs(data_folder = os.path.join("input_data", "paragraphs"), paragraph_text_file = os.path.join("input_data", "paragraph_texts.csv"))