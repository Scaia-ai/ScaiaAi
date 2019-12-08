import gensim
import os
import pandas as pd


def read_corpus(df, tokens_only=False):
    for i, doc in zip(df["filename"], df["text"]):
        if tokens_only:
            yield gensim.utils.simple_preprocess(str(doc))
        else:
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(str(doc)), [i])


def train_doc2vec(model_file_name = os.path.join("input_data", "doc2vec_model.bin"), embeddings_file_path = os.path.join("input_data", "email_embeddings.csv")):

    df      = pd.read_csv(embeddings_file_path, usecols=["filename", "text"])
    corpus  = list(read_corpus(df))

    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(corpus) # build vocabulary
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(model_file_name)

    print("Model saved to: {}".format(model_file_name))
    return


def test_doc2vec(embeddings_file_path = os.path.join("input_data", "email_embeddings.csv"),
                 model_file_name = os.path.join("input_data", "doc2vec_model.bin"),
                 results_path = os.path.join("output_data", "doc2vec_testing.csv")):

    df      = pd.read_csv(embeddings_file_path, usecols=["filename", "text"])
    corpus  = list(read_corpus(df))

    model = gensim.models.doc2vec.Doc2Vec.load(model_file_name)

    results_df = pd.DataFrame(columns=["doc_index", "is_same_doc_closest", "total_num_closest_docs"])
    results_df.to_csv(results_path, index=False)
    for document in corpus:
        doc     = document.words
        doc_ix  = document.tags[0]

        inferred_vector = model.infer_vector(doc)
        sims            = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

        total_num_closest_docs = 0
        for ix, prediction in enumerate(sims):
            if ix == 0:
                is_same_doc_closest = doc_ix == prediction[0]
            if prediction[1] > 0.7:
                total_num_closest_docs +=1
            else:
                break

        tmp_df = pd.DataFrame(columns=["doc_index", "is_same_doc_closest", "total_num_closest_docs"])
        tmp_df.loc[0] = [doc_ix, is_same_doc_closest, total_num_closest_docs]
        with open(results_path, 'a', encoding="utf-8") as pf:
            tmp_df.to_csv(pf, header=False, index=False)

    print("Results saved to: {}".format(model_file_name))
    return

def predict_doc2vec(single_doc, model_file_name = os.path.join("input_data", "doc2vec_model.bin"), closest_docs_threshold = 4):

    df = pd.DataFrame([[0, single_doc]], columns=["filename", "text"])

    corpus  = list(read_corpus(df))

    model = gensim.models.doc2vec.Doc2Vec.load(model_file_name)

    document = corpus[0]
    doc     = document.words
    # doc_ix  = document.tags[0]

    inferred_vector = model.infer_vector(doc)
    sims            = model.docvecs.most_similar([inferred_vector], topn=closest_docs_threshold)

    return sims


if __name__ == "__main__":

    single_doc = "2 of our counterparties are writing letters of complaint.. here's a sample of some of the quotes we have heard from the 10 counterparties we have added in the last 6 months. "
    print(predict_doc2vec(single_doc, model_file_name = os.path.join("input_data", "doc2vec_model.bin"), closest_docs_threshold = 4))
