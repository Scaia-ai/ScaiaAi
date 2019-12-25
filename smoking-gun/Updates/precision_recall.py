import os
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score


def p_r_f1_scores(man_results_path       = os.path.join("output_data", "bert_testing_manhattan.csv"),
                  cos_results_path        = os.path.join("output_data", "bert_testing_cosine.csv"),
                  doc2vec_results_path    = os.path.join("output_data", "doc2vec_testing.csv")):
    for model_name, file_path in [("bert_manhattan", man_results_path), ("bert_cosine", cos_results_path), ("doc2vec", doc2vec_results_path)]:
        print("\nModel Name: ", model_name)
        test_df = pd.read_csv(file_path)
        print("Precision: ", model_name, precision_score(y_true=test_df["is_same_doc_closest"].apply(lambda x: True), y_pred=test_df["is_same_doc_closest"].astype(bool)))
        print("Recall: ", model_name, recall_score(y_true=test_df["is_same_doc_closest"].apply(lambda x: True), y_pred=test_df["is_same_doc_closest"].astype(bool)))
        print("F1 score: ", model_name, f1_score(y_true=test_df["is_same_doc_closest"].apply(lambda x: True), y_pred=test_df["is_same_doc_closest"].astype(bool)))
    return


if __name__ == "__main__":

    man_results_path = os.path.join("output_data", "bert_testing_manhattan.csv")
    cos_results_path = os.path.join("output_data", "bert_testing_cosine.csv")
    doc2vec_results_path = os.path.join("output_data", "doc2vec_testing.csv")

    p_r_f1_scores()
