from sklearn.neighbors import NearestNeighbors
import os, ast, re, joblib
import pandas as pd
import numpy as np
from bert_embeddings import get_single_doc_bert_embedding


def get_closest_docs(document_embedding, df, neighbors_model, top_n_results):

    distances, indices = neighbors_model.kneighbors([document_embedding], n_neighbors=top_n_results, return_distance=True)

    results = {}
    for ix, distance_index in enumerate(zip(list(distances[0]), list(indices[0]))):
        distance, index = distance_index
        results[ix+1]   = {"distance": round(distance, 3),
                           "doc_index": index,
                           "text": df["text"][index]}

    return results


def train_nn(embeddings_file_path = os.path.join("input_data", "mueller_embeddings.csv"), distance_type="cosine", file_prefix=""):
    output_model_path = os.path.join("input_data", "{}_{}_model.pkl".format(file_prefix, distance_type))

    df = pd.read_csv(embeddings_file_path)

    df["embedding"] = df["embedding"].apply(lambda x: ast.literal_eval(re.sub("\s+", ", ", re.sub("\[\s+", "[", x))))
    #train_size = int(len(df) * .80)
    #df = df.iloc[:train_size]
    neighbors = NearestNeighbors(n_neighbors=1, metric=distance_type)
    neighbors.fit(list(df["embedding"]))

    joblib.dump(neighbors, output_model_path)

    print("Model saved to: {}".format(output_model_path))

    return


def test_nn(embeddings_file_path = os.path.join("input_data", "muller_embeddings.csv"), distance_type="cosine", file_prefix=""):

    model_path = os.path.join("input_data", "{}_{}_model.pkl".format(file_prefix, distance_type))
    
    if not os.path.isdir("output_data"):
        os.makedirs("output_data")
    
    results_path = os.path.join("output_data", "{}_bert_testing_{}.csv".format(file_prefix, distance_type))

    df              = pd.read_csv(embeddings_file_path)
    
    
    df["embedding"] = df["embedding"].apply(lambda x: ast.literal_eval(re.sub("\s+", ", ", re.sub("\[\s+", "[", x))))

    neighbors       = joblib.load(model_path)
    #test_size = int(len(df) * .20)
    #df = df.iloc[-test_size:]
    #df = df.reset_index()
    
    datasize = len(df)
    results_df = pd.DataFrame(columns=["doc_index", "is_same_doc_closest", "total_num_closest_docs"])
    results_df.to_csv(results_path, index=False)
    for test_sample_index in range(datasize):
        print(test_sample_index, "/", datasize)

        results = get_closest_docs(document_embedding=df["embedding"][test_sample_index], df=df, neighbors_model=neighbors, top_n_results=datasize)
        tmp_df  = pd.DataFrame.from_dict(results, orient="index")
        tmp_df  = tmp_df[tmp_df["distance"]==0.0]

        num_closest_documents   = len(tmp_df)
        same_doc_present        = test_sample_index in tmp_df["doc_index"].values

        tmp_df = pd.DataFrame(columns=["doc_index", "is_same_doc_closest", "total_num_closest_docs"])
        tmp_df.loc[0] = [test_sample_index, same_doc_present, num_closest_documents]
        with open(results_path, 'a', encoding="utf-8") as pf:
            tmp_df.to_csv(pf, header=False, index=False)

    print("Results saved to: {}".format(results_path))
    return


def predict_nn(document, embeddings_df, tokenizer, model, closest_docs_threshold = 4, distance_type="cosine", file_prefix=""):

    embeddings = get_single_doc_bert_embedding(document, tokenizer, model)

    # model_path = os.path.join("input_data", "{}_model_10000.pkl".format(distance_type))
    model_path = os.path.join("input_data", "{}_{}_model.pkl".format(file_prefix, distance_type))

    # and later you can load it
    neighbors = joblib.load(model_path)

    return get_closest_docs(document_embedding=embeddings, df=embeddings_df, neighbors_model=neighbors, top_n_results=closest_docs_threshold)


if __name__ == "__main__":

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

    document = "Collusion with Russia. "

    predict_nn(document, df, tokenizer, model, closest_docs_threshold=4, distance_type="cosine")
