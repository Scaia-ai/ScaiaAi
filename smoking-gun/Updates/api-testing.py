#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Request-Url" data-toc-modified-id="Request-Url-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Request Url</a></span></li><li><span><a href="#Case-Id's-(caseUUID)-for-Datasets-available" data-toc-modified-id="Case-Id's-(caseUUID)-for-Datasets-available-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Case Id's (caseUUID) for Datasets available</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#&quot;CUUID001&quot;:&quot;freed&quot;" data-toc-modified-id="&quot;CUUID001&quot;:&quot;freed&quot;-2.0.1"><span class="toc-item-num">2.0.1&nbsp;&nbsp;</span>"CUUID001":"freed"</a></span></li><li><span><a href="#&quot;CUUID002&quot;:&quot;enron&quot;" data-toc-modified-id="&quot;CUUID002&quot;:&quot;enron&quot;-2.0.2"><span class="toc-item-num">2.0.2&nbsp;&nbsp;</span>"CUUID002":"enron"</a></span></li></ul></li></ul></li><li><span><a href="#Request-Format" data-toc-modified-id="Request-Format-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Request Format</a></span></li><li><span><a href="#All-data-&amp;-models-sample-testing" data-toc-modified-id="All-data-&amp;-models-sample-testing-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>All data &amp; models sample testing</a></span><ul class="toc-item"><li><span><a href="#Freed---Doc2vec-Test" data-toc-modified-id="Freed---Doc2vec-Test-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Freed - Doc2vec Test</a></span></li><li><span><a href="#Freed---Bert-Jaccard-Test" data-toc-modified-id="Freed---Bert-Jaccard-Test-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Freed - Bert-Jaccard Test</a></span></li><li><span><a href="#Freed---Bert-Cosine-Test" data-toc-modified-id="Freed---Bert-Cosine-Test-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Freed - Bert-Cosine Test</a></span></li><li><span><a href="#Enron---Doc2vec-Test" data-toc-modified-id="Enron---Doc2vec-Test-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Enron - Doc2vec Test</a></span></li><li><span><a href="#Enron---Bert-Jaccard-Test" data-toc-modified-id="Enron---Bert-Jaccard-Test-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Enron - Bert-Jaccard Test</a></span></li><li><span><a href="#Enron---Bert-Cosine-Test" data-toc-modified-id="Enron---Bert-Cosine-Test-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Enron - Bert-Cosine Test</a></span></li></ul></li></ul></div>

# # Request Url

# In[1]:


import json, socket, requests


# In[2]:


base_url    = 'http://3.14.125.217:12345/aiAdvisor'

headers = {"Content-Type": "application/json"}


# # Case Id's (caseUUID) for Datasets available

# ### "CUUID001":"freed" 
#     for usage of freed dataset
# ### "CUUID002":"enron" 
#     for usage of enron dataset

# # Request Format
#     {"caseUUID":"<CUUID###>",
#     "modelType": str, # available = doc2vec, bertJaccard, bertCosine
#     "threshold": int,
#     "document":"<search query>"}

# # All data & models sample testing

# ## Freed - Doc2vec Test

# In[18]:


input_json = {"caseUUID":"CUUID001",
              "modelType": "doc2vec",
              "threshold": 4,
              "document":"2 of our counterparties are writing letters of complaint.. here's a sample of some of the quotes we have heard from the 10 counterparties we have added in the last 6 months. "}

payload = json.dumps(input_json)

r = requests.post(base_url, headers=headers, data=payload)
print(json.dumps(json.loads(r.content), indent=4))


# ## Freed - Bert-Jaccard Test

# In[19]:


input_json = {"caseUUID":"CUUID001",
              "modelType": "bertJaccard",
              "threshold": 4,
              "document":"2 of our counterparties are writing letters of complaint.. here's a sample of some of the quotes we have heard from the 10 counterparties we have added in the last 6 months. "}

payload = json.dumps(input_json)

r = requests.post(base_url, headers=headers, data=payload)
print(json.dumps(json.loads(r.content), indent=4))


# ## Freed - Bert-Cosine Test

# In[21]:


input_json = {"caseUUID":"CUUID001",
              "modelType": "bertCosine",
              "threshold": 4,
              "document":"2 of our counterparties are writing letters of complaint.. here's a sample of some of the quotes we have heard from the 10 counterparties we have added in the last 6 months. "}

payload = json.dumps(input_json)

r = requests.post(base_url, headers=headers, data=payload)
print(json.dumps(json.loads(r.content), indent=4))


# In[ ]:





# In[ ]:





# In[ ]:





# ## Enron - Doc2vec Test

# In[22]:


input_json = {"caseUUID":"CUUID002",
              "modelType": "doc2vec",
              "threshold": 4,
              "document":"2 of our counterparties are writing letters of complaint.. here's a sample of some of the quotes we have heard from the 10 counterparties we have added in the last 6 months. "}

payload = json.dumps(input_json)

r = requests.post(base_url, headers=headers, data=payload)
print(json.dumps(json.loads(r.content), indent=4))


# ## Enron - Bert-Manhattan Test

# In[23]:


input_json = {"caseUUID":"CUUID002",
              "modelType": "bertManhattan",
              "threshold": 4,
              "document":"2 of our counterparties are writing letters of complaint.. here's a sample of some of the quotes we have heard from the 10 counterparties we have added in the last 6 months. "}

payload = json.dumps(input_json)

r = requests.post(base_url, headers=headers, data=payload)
print(json.dumps(json.loads(r.content), indent=4))


# ## Enron - Bert-Cosine Test

# In[25]:


input_json = {"caseUUID":"CUUID002",
              "modelType": "bertCosine",
              "threshold": 4,
              "document":"2 of our counterparties are writing letters of complaint.. here's a sample of some of the quotes we have heard from the 10 counterparties we have added in the last 6 months. "}

payload = json.dumps(input_json)

r = requests.post(base_url, headers=headers, data=payload)
print(json.dumps(json.loads(r.content), indent=4))


# In[ ]:




