# ####################################################################
# to process documents (i.e, emails & save them into csv)
# ####################################################################

import os, re
import pandas as pd


def process_emails(data_folder = os.path.join("input_data", "emails"), email_text_file = os.path.join("input_data", "email_texts.csv"),
                  text_filter=""):

    df = pd.DataFrame(columns=["filename", "text"])
    df.to_csv(email_text_file, index=False)


    for ix, fn in enumerate(os.listdir(data_folder)):

        with open(os.path.join(data_folder, fn), "r", encoding="utf-8") as fp:
            email_text = fp.readlines()

        # cleaning text of newlines and strings with only special characters
        email_text = " ".join(filter(lambda x: re.sub('[^A-Za-z0-9]+', '', x.replace(text_filter, "")), email_text))

        if email_text.strip()=="":
            continue
        
        tmp_df = pd.DataFrame(columns=["filename", "text"])
        tmp_df.loc[0] = [fn, email_text]
        with open(email_text_file, 'a', encoding="utf-8") as pf:
            tmp_df.to_csv(pf, header=False, index=False)

    return

if __name__ == "__main__":
    process_emails(data_folder = os.path.join("input_data", "emails"), email_text_file = os.path.join("input_data", "email_texts.csv"))