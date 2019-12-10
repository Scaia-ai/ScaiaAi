#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#To-download-Test-Data-from-github" data-toc-modified-id="To-download-Test-Data-from-github-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>To download Tesla Docs from github</a></span></li><li><span><a href="#Process-Tesla-Docs" data-toc-modified-id="Process-Tesla-Docs-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Process Tesla Docs</a></span></li><li><span><a href="#Create-Bert-Embeddings-for-Tesla-Docs" data-toc-modified-id="Create-Bert-Embeddings-for-Tesla-Docs-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Create Bert Embeddings for Tesla Docs</a></span></li><li><span><a href="#Nearest-Neighbors---Jaccard" data-toc-modified-id="Nearest-Neighbors---Jaccard-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Nearest Neighbors - Jaccard</a></span></li><li><span><a href="#Nearest-Neighbors---Cosine" data-toc-modified-id="Nearest-Neighbors---Cosine-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Nearest Neighbors - Cosine</a></span></li><li><span><a href="#Doc2Vec" data-toc-modified-id="Doc2Vec-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Doc2Vec</a></span></li><li><span><a href="#Precision-&amp;-Recall" data-toc-modified-id="Precision-&amp;-Recall-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Precision &amp; Recall</a></span></li></ul></div>

# #  To download Test Data from github

# In[ ]:

import  os

from get_test_data import clone_test_data


# In[ ]:


clone_test_data(download_to_folder = os.path.join("input_data", "emails"))


# # Process Telsa Docs

# In[ ]:


import os
from process_data import process_emails


# In[ ]:


process_data(data_folder = os.path.join("input_data", "emails"), email_text_file = os.path.join("input_data", "enron_texts.csv"))


# # Create Bert Embeddings for Telsa

# In[ ]:


from document_representations_bert import get_doc_representations


# In[ ]:


get_doc_representations(input_email_text_file=os.path.join("input_data", "enron_texts.csv"),
                        output_embedding_file=os.path.join("input_data", "enron_embeddings.csv"))


# 

# # Nearest Neighbors - Jaccard

# In[ ]:


import os
from nn_similarity import train_nn, test_nn


# In[ ]:


# train model
train_nn(embeddings_file_path = os.path.join("input_data", "enron_embeddings.csv"), 
         distance_type="jaccard", 
         file_prefix="enron")


# In[ ]:


# test model on same train data
test_nn(embeddings_file_path = os.path.join("input_data", "enron_embeddings.csv"), 
        distance_type="jaccard", 
        file_prefix="enron")


# # Nearest Neighbors - Cosine

# In[ ]:


import os
from nn_similarity import train_nn, test_nn


# In[ ]:


# train model
train_nn(embeddings_file_path = os.path.join("input_data", "enron_embeddings.csv"), 
         distance_type="cosine", 
         file_prefix="enron")


# In[ ]:


# test model on same train data
test_nn(embeddings_file_path = os.path.join("input_data", "enron_embeddings.csv"), 
        distance_type="cosine",
       file_prefix="enron")


# # Doc2Vec

# In[ ]:


import os
from doc2vec_similarity import train_doc2vec, test_doc2vec


# In[ ]:


train_doc2vec(model_file_name = os.path.join("input_data", "enron_doc2vec_model.bin"), 
              embeddings_file_path = os.path.join("input_data", "enron_embeddings.csv"))


# In[ ]:


test_doc2vec(embeddings_file_path = os.path.join("input_data", "enron_embeddings.csv"), 
             model_file_name = os.path.join("input_data", "enron_doc2vec_model.bin"),
             results_path = os.path.join("output_data", "enron_doc2vec_testing.csv"))


# # Precision & Recall

# In[ ]:


import os
from precision_recall import p_r_f1_scores


# In[ ]:


p_r_f1_scores(jacc_results_path       = os.path.join("output_data", "enron_bert_testing_jaccard.csv"),
              cos_results_path        = os.path.join("output_data", "enron_bert_testing_cosine.csv"),
              doc2vec_results_path    = os.path.join("output_data", "enron_doc2vec_testing.csv"))


# In[ ]:




