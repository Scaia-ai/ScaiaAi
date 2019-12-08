#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#To-download-Enron-Data-from-Web" data-toc-modified-id="To-download-Enron-Data-from-Web-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>To download Enron Data from Web</a></span></li><li><span><a href="#Process-Emails" data-toc-modified-id="Process-Emails-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Process Emails</a></span></li><li><span><a href="#Create-Bert-Embeddings-for-Emails" data-toc-modified-id="Create-Bert-Embeddings-for-Emails-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Create Bert Embeddings for Emails</a></span></li><li><span><a href="#Nearest-Neighbors---Jaccard" data-toc-modified-id="Nearest-Neighbors---Jaccard-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Nearest Neighbors - Jaccard</a></span></li><li><span><a href="#Nearest-Neighbors---Cosine" data-toc-modified-id="Nearest-Neighbors---Cosine-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Nearest Neighbors - Cosine</a></span></li><li><span><a href="#Doc2Vec" data-toc-modified-id="Doc2Vec-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Doc2Vec</a></span></li><li><span><a href="#Precision-&amp;-Recall" data-toc-modified-id="Precision-&amp;-Recall-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Precision &amp; Recall</a></span></li></ul></div>

# # To download Enron Data from Web

# In[1]:


from get_enron import download_enron_data


# In[2]:


download_enron_data(download_to_folder = os.path.join("input_data", "emails"))


# # Process Emails

# In[6]:


import os
from process_data import process_emails


# In[7]:


process_emails(data_folder = os.path.join("input_data", "emails"), email_text_file = os.path.join("input_data", "enron_texts.csv"))


# # Create Bert Embeddings for Emails

# In[8]:


from document_representations_bert import get_doc_representations


# In[9]:


get_doc_representations(input_email_text_file=os.path.join("input_data", "enron_texts.csv"),
                        output_embedding_file=os.path.join("input_data", "enron_embeddings.csv"))


# Stopped above line after a while, as there are 1043264 records and 124 records per chunk, we would need to process 8414 times, and this would take more than a week to process, hence stopped. We still have good amount of data, i.e., 247*124 = 30628

# # Nearest Neighbors - Jaccard

# In[1]:


import os
from nn_similarity import train_nn, test_nn


# In[2]:


# train model
train_nn(embeddings_file_path = os.path.join("input_data", "enron_embeddings.csv"), 
         distance_type="jaccard", 
         file_prefix="enron")


# In[5]:


# test model on same train data
test_nn(embeddings_file_path = os.path.join("input_data", "enron_embeddings.csv"), 
        distance_type="jaccard", 
        file_prefix="enron")


# # Nearest Neighbors - Cosine

# In[2]:


import os
from nn_similarity import train_nn, test_nn


# In[7]:


# train model
train_nn(embeddings_file_path = os.path.join("input_data", "enron_embeddings.csv"), 
         distance_type="cosine", 
         file_prefix="enron")


# In[3]:


# test model on same train data
test_nn(embeddings_file_path = os.path.join("input_data", "enron_embeddings.csv"), 
        distance_type="cosine",
       file_prefix="enron")


# # Doc2Vec

# In[4]:


import os
from doc2vec_similarity import train_doc2vec, test_doc2vec


# In[2]:


train_doc2vec(model_file_name = os.path.join("input_data", "enron_doc2vec_model.bin"), 
              embeddings_file_path = os.path.join("input_data", "enron_embeddings.csv"))


# In[5]:


test_doc2vec(embeddings_file_path = os.path.join("input_data", "enron_embeddings.csv"), 
             model_file_name = os.path.join("input_data", "enron_doc2vec_model.bin"),
             results_path = os.path.join("output_data", "enron_doc2vec_testing.csv"))


# # Precision & Recall

# In[8]:


import os
from precision_recall import p_r_f1_scores


# In[10]:


p_r_f1_scores(jacc_results_path       = os.path.join("output_data", "enron_bert_testing_jaccard.csv"),
              cos_results_path        = os.path.join("output_data", "enron_bert_testing_cosine.csv"),
              doc2vec_results_path    = os.path.join("output_data", "enron_doc2vec_testing.csv"))


# In[ ]:




