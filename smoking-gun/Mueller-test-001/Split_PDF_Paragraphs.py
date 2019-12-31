import os
import os.path as op
from tqdm import tqdm
import re 

import numpy as np
import pandas as pd


# Directory where the input .pdf is located
in_path = 'Mueller-report.pdf'

# Output is the path where the split files will be saved
out_path = 'input_data/paragraphs'


# Create output directory if it doesn't exist

if not op.isdir(out_path):
    os.makedirs(out_path)

# Function for converting pdf to a string. It does this by reading in each page of the pdf 
# and converting to string. The final output is a string of all pages. 
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text


processed = convert_pdf_to_txt(in_path)


# ## Split processed text by new lines to get list of paragraphs


paragraphs = processed.split("\n\n \n\n \n\n")


# In[9]:


paragraphs


# In[10]:


len(paragraphs)


# ## Iterate through list of paragraphs and write into text files in paragraphs directory 

# In[11]:


# For loop splits by paragraph and outputs each one into a new text file 

cnt = 0

for i in range(len(paragraphs)):
    if len(paragraphs[i]) < 10:
        continue
    else:
        out = open(op.join(out_path, 'paragraph_{:04d}.txt'.format(cnt)), "w+")
        out.writelines(paragraphs[i])
        out.close()
        cnt += 1
    


# In[ ]:




