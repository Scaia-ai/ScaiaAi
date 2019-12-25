#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Process-Emails" data-toc-modified-id="Process-Emails-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Process Emails</a></span></li><li><span><a href="#Create-Bert-Embeddings-for-Emails" data-toc-modified-id="Create-Bert-Embeddings-for-Emails-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Create Bert Embeddings for Emails</a></span></li><li><span><a href="#Nearest-Neighbors---Jaccard" data-toc-modified-id="Nearest-Neighbors---Jaccard-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Nearest Neighbors - Jaccard</a></span></li><li><span><a href="#Nearest-Neighbors---Cosine" data-toc-modified-id="Nearest-Neighbors---Cosine-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Nearest Neighbors - Cosine</a></span></li><li><span><a href="#Doc2Vec" data-toc-modified-id="Doc2Vec-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Doc2Vec</a></span></li><li><span><a href="#Precision-&amp;-Recall" data-toc-modified-id="Precision-&amp;-Recall-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Precision &amp; Recall</a></span></li></ul></div>

# # Process Emails

# In[2]:


import os

from process_data import process_emails


# In[3]:


process_emails(data_folder = os.path.join("input_data", "results", "text"), 
               email_text_file = os.path.join("input_data", "freed_texts.csv"),
               text_filter="EDRM Enron Email Data Set has been produced in EML, PST and NSF format by ZL Technologies, Inc. This Data Set is licensed under a Creative Commons Attribution 3.0 United States License <http://creativecommons.org/licenses/by/3.0/us/> . To provide attribution, please cite to \"ZL Technologies, Inc. (http://www.zlti.com)")


# # Create Bert Embeddings for Emails

# In[6]:


from document_representations_bert import get_doc_representations


# In[7]:


get_doc_representations(input_email_text_file=os.path.join("input_data", "freed_texts.csv"),
                        output_embedding_file=os.path.join("input_data", "freed_embeddings.csv"))


# # Nearest Neighbors - Jaccard

# In[8]:


import os
from nn_similarity import train_nn, test_nn


# In[9]:


# train model
train_nn(embeddings_file_path = os.path.join("input_data", "freed_embeddings.csv"), 
         distance_type="jaccard", 
         file_prefix="freed")


# In[10]:


# test model on same train data
test_nn(embeddings_file_path = os.path.join("input_data", "freed_embeddings.csv"), 
        distance_type="jaccard", 
        file_prefix="freed")


# # Nearest Neighbors - Cosine

# In[11]:


import os
from nn_similarity import train_nn, test_nn


# In[12]:


# train model
train_nn(embeddings_file_path = os.path.join("input_data", "freed_embeddings.csv"), 
         distance_type="cosine",
         file_prefix="freed")


# In[13]:


# test model on same train data
test_nn(embeddings_file_path = os.path.join("input_data", "freed_embeddings.csv"), 
        distance_type="cosine",
        file_prefix="freed")


# # Doc2Vec

# In[14]:


import os
from doc2vec_similarity import train_doc2vec, test_doc2vec


# In[15]:


train_doc2vec(model_file_name = os.path.join("input_data", "freed_doc2vec_model.bin"), 
              embeddings_file_path = os.path.join("input_data", "freed_embeddings.csv"))


# In[16]:


test_doc2vec(embeddings_file_path = os.path.join("input_data", "freed_embeddings.csv"), 
             model_file_name = os.path.join("input_data", "freed_doc2vec_model.bin"),
             results_path = os.path.join("output_data", "freed_doc2vec_testing.csv"))


# # Precision & Recall

# In[17]:


import os
from precision_recall import p_r_f1_scores


# In[18]:


p_r_f1_scores(man_results_path      = os.path.join("output_data", "freed_bert_testing_jaccard.csv"),
              cos_results_path        = os.path.join("output_data", "freed_bert_testing_cosine.csv"),
              doc2vec_results_path    = os.path.join("output_data", "freed_doc2vec_testing.csv"))


# In[ ]:





# In[ ]:




