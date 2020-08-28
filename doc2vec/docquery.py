#!/usr/bin/env python
# coding: utf-8

# # Doc2Vec Tutorial on the Lee Dataset

# In[1]:


import gensim
import os
import collections
import smart_open
import random
import argparse
import re
import sys

from gensim.test.utils import get_tmpfile


fname = "model.bin"

inputfile_dir = "./input_data/emails"

creative_commons = "EDRM Enron Email Data Set has been produced in EML, PST and NSF format by ZL Technologies, Inc. This Data Set is licensed under a Creative Commons Attribution 3.0 United States License <http://creativecommons.org/licenses/by/3.0/us/> . To provide attribution, please cite to \"ZL Technologies, Inc. (http://www.zlti.com).\""

enron_disclaimer = """This e-mail is the property of Enron Corp. and/or its relevant affiliate and
may contain confidential and privileged material for the sole use of the
intended recipient (s). Any review, use, distribution or disclosure by
others is strictly prohibited. If you are not the intended recipient (or
authorized to receive for the recipient), please contact the sender or reply
to Enron Corp. at enron.messaging.administration@enron.com and delete all
copies of the message. This e-mail (and any attachments hereto) are not
intended to be an offer (or an acceptance) and do not create or evidence a
binding and enforceable contract between Enron Corp. (or any of its
affiliates) and the intended recipient or any other party, and may not be
relied on by anyone as the basis of a contract by estoppel or otherwise.
Thank you."""




def read_corpus_dir(dirname):
    return_list = []
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            with open(dirname + '/' + fname, 'r') as file:
                doc = file.read()
                if creative_commons in doc:
                    doc = doc.replace(creative_commons, '')
                if doc.strip() and len(doc.strip()) > 50: # Not empty and length > 50 characters
                    return_list.append(doc)
    return return_list

def read_corpus(corpus_list, tokens_only=False):
    for i, doc in enumerate(corpus_list):
        if tokens_only:
            yield gensim.utils.simple_preprocess(doc)
        else:
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(doc), [i])

def read_doc(filename):
    return_doc = ""
    with open(filename, 'r') as file:
        doc = file.read()
        if creative_commons in doc:
            doc = doc.replace(creative_commons, '')
            doc = doc.replace(enron_disclaimer, '')
        if doc.strip() and len(doc.strip()) > 50: # Not empty and length > 50 characters
            return_doc = doc

    return gensim.utils.simple_preprocess(return_doc)
                

corpus = read_corpus_dir(inputfile_dir)

print("Found %d files in %s" % (len(corpus), inputfile_dir))

final_corpus = list(read_corpus(corpus))

train_corpus = final_corpus

test_corpus = list(read_corpus(corpus[:500], tokens_only=True))



# ## Training the Model

# ### Instantiate a Doc2Vec Object 

# Now, we'll instantiate a Doc2Vec model with a vector size with 50 words and iterating over the training corpus 40 times. We set the minimum word count to 2 in order to discard words with very few occurrences. (Without a variety of representative examples, retaining such infrequent words can often make a model worse!) Typical iteration counts in published 'Paragraph Vectors' results, using 10s-of-thousands to millions of docs, are 10-20. More iterations take more time and eventually reach a point of diminishing returns.

model = None

if os.path.exists(fname): # if model already exists, don't retrain
  print("found existing model, loading...")
  model =  gensim.models.doc2vec.Doc2Vec.load(fname)

else: # otherwise, retrain
  print("retraining...")
  model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

  # ### Build a Vocabulary
  # Essentially, the vocabulary is a dictionary (accessible via `model.wv.vocab`) of all of the unique words extracted from the training corpus along with the count (e.g., `model.wv.vocab['penalty'].count` for counts for the word `penalty`).

  model.build_vocab(train_corpus) # build vocabulary
  model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
  model.save(fname)


## Get value of input document

document_to_read = [' '.join(sys.argv[1:])]
print(document_to_read)
inferred_vector = model.infer_vector(document_to_read)
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document: «{}»\n'.format(' '.join(document_to_read)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('TOP', 0), ('Second', 1), ('Third', 2), ('Fourth', 3), ('Fifth', 4), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))









