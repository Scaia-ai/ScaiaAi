import os
import datetime
import sys

import pandas as pd

try:
    aDate = datetime.date.fromisoformat('2019-12-04')
    print("good date")
    print(aDate)
except (Exception):
    print(Exception)

sys.exit()

nagios_prefix = 'nagios-'
crystal_prefix = 'Crystal-'
decrystal_prefix = 'Decrys

file_path = '/home/mark/projects/scaia/scaia-test-data/jeff/metadata_0.csv'
file_name = os.path.basename(file_path)

df = pd.read_csv(file_path, sep='|')

if (os.path.exists("output") == False):
    os.mkdir("output")

def dater(src_text):
    if not src_text or src_text == "" or not isinstance(src_text, str):
        return ("")
    if (src_text.startswith(nagios_prefix)):
        src_text = src_text[len(nagios_prefix):]
    if (src_text.startswith(crystal_prefix)):
        src_text = src_text[len(crystal_prefix):]
    if (src_text.startswith(decrystal_prefix)):
        src_text = src_text[len(decrystal_prefix):]

    my_date = src_text[:10]
    try:
        aDate = datetime.date.fromisoformat(my_date)
        print("good date")
        print(aDate)
    except (Exception):
        # print("bad date")
        my_date = ""

    return(my_date)

df['file_date'] = df['Source Path'].apply(dater)
df['tag'] = "whatever"

# Output final results
df[['tag', 'file_date']].to_csv('output/' + file_name + '_tags.csv', sep='|', index=False)