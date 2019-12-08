import logging
import os
import sys
import re
import tarfile
import itertools

import nltk
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

def process_message(message):
    """
    Preprocess a single 20newsgroups message, returning the result as
    a unicode string.
    
    """
    message = gensim.utils.to_unicode(message, 'latin1').strip()
    blocks = message.split(u'\n\n')
    # skip email headers (first block) and footer (last block)
    content = u'\n\n'.join(blocks[1:])
    return content


with tarfile.open('./data/20news-bydate.tar.gz', 'r:gz') as tf:
    # get information (metadata) about all files in the tarball
    file_infos = [file_info for file_info in tf if file_info.isfile()]
    
    # print one of them; for example, the first one
    message = tf.extractfile(file_infos[0]).read()
    documents = [ process_message(tf.extractfile(file_infos[i]).read()) for i in range(len(file_infos)) ]
    print(documents[0])

print('---')
print(process_message(message))



def iter_20newsgroups(fname, log_every=None):
    """
    Yield plain text of each 20 newsgroups message, as a unicode string.

    The messages are read from raw tar.gz file `fname` on disk (e.g. `./data/20news-bydate.tar.gz`)

    """
    extracted = 0
    with tarfile.open(fname, 'r:gz') as tf:
        for file_number, file_info in enumerate(tf):
            if file_info.isfile():
                if log_every and extracted % log_every == 0:
                    logging.info("extracting 20newsgroups file #%i: %s" % (extracted, file_info.name))
                content = tf.extractfile(file_info).read()
                yield process_message(content)
                extracted += 1

# itertools is an inseparable friend with data streaming (Python built-in library)
import itertools

# let's only parse and print the first three messages, lazily
# `list(stream)` materializes the stream elements into a plain Python list
message_stream = iter_20newsgroups('./data/20news-bydate.tar.gz', log_every=2)
print(list(itertools.islice(message_stream, 3)))


class Corpus20News(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for text in iter_20newsgroups(self.fname):
            # tokenize each message; simply lowercase & match alphabetic chars, for now
            yield list(gensim.utils.tokenize(text, lower=True))

tokenized_corpus = Corpus20News('./data/20news-bydate.tar.gz')

# print the first two tokenized messages
print(list(itertools.islice(tokenized_corpus, 2)))


dictionary = corpora.Dictionary(tokenized_corpus)
corpus = [dictionary.doc2bow(text) for text in tokenized_corpus]

print(dictionary[0])
print(corpus[0])


# Matrix similarity

from gensim import models

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

from gensim import similarities
index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it


index.save('/tmp/20news.index')
index = similarities.MatrixSimilarity.load('/tmp/20news.index')



# Run a query 
test_doc = "Human computer interaction"
test_doc = input("Enter some text: " )
test_vec_bow = dictionary.doc2bow(test_doc.lower().split())
test_vec_lsi = lsi[test_vec_bow]  # convert the query to LSI space
print(test_vec_lsi)



sims = index[test_vec_lsi]  # perform a similarity query against the corpus
sims = sorted(enumerate(sims), key=lambda item: -item[1])

print(sims)

print("------------------------------------------")

for i, s in enumerate(sims[:10]):
    print("------------------------------------------")
    print("Document number " + str(i) + ":")
    print(s, documents[i])
    if i > 10:
        break
    print("------------------------------------------")

