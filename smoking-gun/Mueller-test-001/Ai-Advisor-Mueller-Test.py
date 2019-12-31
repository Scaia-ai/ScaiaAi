#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#To-download-Enron-Data-from-Web" data-toc-modified-id="To-download-Enron-Data-from-Web-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>To download Enron Data from Web</a></span></li><li><span><a href="#Process-Emails" data-toc-modified-id="Process-Emails-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Process Emails</a></span></li><li><span><a href="#Create-Bert-Embeddings-for-Emails" data-toc-modified-id="Create-Bert-Embeddings-for-Emails-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Create Bert Embeddings for Emails</a></span></li><li><span><a href="#Nearest-Neighbors---Jaccard" data-toc-modified-id="Nearest-Neighbors---Jaccard-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Nearest Neighbors - Jaccard</a></span></li><li><span><a href="#Nearest-Neighbors---Cosine" data-toc-modified-id="Nearest-Neighbors---Cosine-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Nearest Neighbors - Cosine</a></span></li><li><span><a href="#Doc2Vec" data-toc-modified-id="Doc2Vec-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Doc2Vec</a></span></li><li><span><a href="#Precision-&amp;-Recall" data-toc-modified-id="Precision-&amp;-Recall-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Precision &amp; Recall</a></span></li></ul></div>

# In[1]:


import os
import ast
import re
import pandas as pd
import scipy


# # Process text files 

# In[2]:


from process_data import process_paragraphs


# In[3]:


process_paragraphs(data_folder = os.path.join("input_data", "paragraphs"),paragraph_text_file = os.path.join("input_data", "mueller_paragraphs.csv"))


# # Create Bert Embeddings for Mueller Paragraphs

# In[4]:


from document_representations_bert import get_doc_representations


# In[5]:


get_doc_representations(input_email_text_file=os.path.join("input_data", "mueller_paragraphs.csv"),
                        output_embedding_file=os.path.join("input_data", "mueller_embeddings.csv"))


# Stopped above line after a while, as there are 1043264 records and 124 records per chunk, we would need to process 8414 times, and this would take more than a week to process, hence stopped. We still have good amount of data, i.e., 247*124 = 30628

# # Nearest Neighbors - Manhattan

# In[6]:


from nn_similarity import train_nn, test_nn, predict_nn


# In[7]:


# train model
train_nn(embeddings_file_path = os.path.join("input_data", "mueller_embeddings.csv"), 
         distance_type="manhattan", 
         file_prefix="mueller")


# In[8]:


# test model on same train data
test_nn(embeddings_file_path = os.path.join("input_data", "mueller_embeddings.csv"), 
        distance_type="manhattan", 
        file_prefix="mueller")


# # Nearest Neighbors - Cosine

# In[9]:


import os
from nn_similarity import train_nn, test_nn


# In[10]:


# train model
train_nn(embeddings_file_path = os.path.join("input_data", "mueller_embeddings.csv"), 
         distance_type="cosine", 
         file_prefix="mueller")


# In[11]:


# test model on same train data
test_nn(embeddings_file_path = os.path.join("input_data", "mueller_embeddings.csv"), 
        distance_type="cosine",
       file_prefix="mueller")


# # Doc2Vec

# In[12]:


from doc2vec_similarity import train_doc2vec, test_doc2vec, predict_doc2vec


# In[13]:


train_doc2vec(model_file_name = os.path.join("input_data", "mueller_doc2vec_model.bin"), 
              embeddings_file_path = os.path.join("input_data", "mueller_embeddings.csv"))


# In[14]:


test_doc2vec(embeddings_file_path = os.path.join("input_data", "mueller_embeddings.csv"), 
             model_file_name = os.path.join("input_data", "mueller_doc2vec_model.bin"),
             results_path = os.path.join("output_data", "mueller_doc2vec_testing.csv"))


# # Precision & Recall

# In[15]:


import os
from precision_recall import p_r_f1_scores


# In[16]:


p_r_f1_scores(man_results_path       = os.path.join("output_data", "mueller_bert_testing_manhattan.csv"),
              cos_results_path        = os.path.join("output_data", "mueller_bert_testing_cosine.csv"),
              doc2vec_results_path    = os.path.join("output_data", "mueller_doc2vec_testing.csv"))


# # Predict closest document to sample

# In[17]:


from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertModel

embeddings_file_path = os.path.join("input_data", "mueller_embeddings.csv")

df = pd.read_csv(embeddings_file_path)
df["embedding"] = df["embedding"].apply(lambda x: ast.literal_eval(re.sub("\s+", ", ", re.sub("\[\s+", "[", x))))

    # Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # # Loading pre-trained model (weights)
    # # and putting the model in "evaluation" mode, meaning feed-forward operation.
model = BertModel.from_pretrained('bert-base-uncased', cache_dir=os.path.join("input_data", "bert"))


# In[22]:


document = "Trump campaign colluded with the Russian government"


# In[23]:


predict_nn(document, df, tokenizer, model, closest_docs_threshold=5, distance_type="manhattan", file_prefix ="mueller")


# In[24]:


predict_nn(document, df, tokenizer, model, closest_docs_threshold=15, distance_type="cosine", file_prefix ="mueller")


# In[25]:


print(predict_doc2vec(document, model_file_name = os.path.join("input_data", "mueller_doc2vec_model.bin"), closest_docs_threshold = 4))


# In[ ]:




