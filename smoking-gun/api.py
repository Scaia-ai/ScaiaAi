import pandas as pd
import json, os, logging, sys, traceback, re, socket, ast
from nn_similarity import predict_nn
from doc2vec_similarity import predict_doc2vec
from pytorch_pretrained_bert import BertTokenizer, BertModel
from flask import Flask, request, Response
from gevent.pywsgi import WSGIServer

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

####################################### Loading Models #######################################


logging.info("loading enron data...")
enron_embeddings_file_path = os.path.join("input_data", "enron_embeddings.csv")
enron_df = pd.read_csv(enron_embeddings_file_path)
enron_df["embedding"] = enron_df["embedding"].apply(lambda x: ast.literal_eval(re.sub("\s+", ", ", re.sub("\[\s+", "[", x))))

logging.info("loading freed data...")
freed_embeddings_file_path = os.path.join("input_data", "freed_embeddings.csv")
freed_df = pd.read_csv(freed_embeddings_file_path)
freed_df["embedding"] = freed_df["embedding"].apply(lambda x: ast.literal_eval(re.sub("\s+", ", ", re.sub("\[\s+", "[", x))))

logging.info("loading bert...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', cache_dir=os.path.join("input_data", "bert"))

case_id_dict = {"CUUID001": "freed",
                "CUUID002": "enron"}

app = Flask(__name__)

def bert_predictions(document, tokenizer, model, closest_docs_threshold, distance_type, file_prefix):
    if file_prefix == "enron":
        df = enron_df
    elif file_prefix == "freed":
        df = freed_df

    predictions = predict_nn(document, df, tokenizer, model, closest_docs_threshold, distance_type, file_prefix)
    for ix in list(predictions.keys()):
        predictions[ix]["doc_index"] = df["filename"][predictions[ix]["doc_index"]]
        if file_prefix == "freed":
            predictions[ix]["uuid"] = predictions[ix]["doc_index"].split(".")[0].split("-")[0]
    return predictions

@app.route('/aiAdvisor', methods=['POST'])
def apiFirst_serve():
    json_data = request.get_json(force=True)

    data_type = case_id_dict[json_data['caseUUID']]

    if data_type not in ["enron", "freed"]:
        predictions = {'check': 'Failed! invalid case uuid'}
        logging.info("Model's OUTPUT:\t" + '\t' + str(predictions))
        resp = Response(json.dumps(predictions))
        resp.headers['Content-type'] = 'application/json'
        return resp

    model_type = json_data['modelType']
    logging.info('Received INPUT:\t' + '\t' + str(json_data))

    if model_type == 'doc2vec':
        document = json_data['document']
        threshold = json_data['threshold']
        if data_type == "enron":
            predictions = predict_doc2vec(document, model_file_name=os.path.join("input_data", "{}_doc2vec_model.bin".format(data_type)), closest_docs_threshold=threshold)
            source_text_path = os.path.join("input_data", "emails")

        elif data_type == "freed":
            predictions = predict_doc2vec(document, model_file_name=os.path.join("input_data", "{}_doc2vec_model.bin".format(data_type)), closest_docs_threshold=threshold)
            source_text_path = os.path.join("input_data", "results", "text")

        results = {}
        for ix, fn_score in enumerate(predictions):
            with open(os.path.join(source_text_path, fn_score[0])) as fn:
                text = "\n".join(fn.readlines())
            results[ix + 1] = {"distance": fn_score[1],
                               "filename": fn_score[0],
                               "text": text}  # distance, filename, text

            if data_type == "freed":
                results["uuid"] = fn_score[0].split(".")[0].split("-")[0]

        predictions = results

    elif model_type == 'bertJaccard':
        document = json_data['document']
        threshold = json_data['threshold']

        predictions = bert_predictions(document, tokenizer, model, threshold, "jaccard", data_type)

    elif model_type == 'bertCosine':
        document = json_data['document']
        threshold = json_data['threshold']
        predictions = bert_predictions(document, tokenizer, model, threshold, "cosine", data_type)

    else:
        predictions = {'check': 'Failed! invalid modelType'}

    logging.info("Model's OUTPUT:\t" + '\t' + str(predictions))

    resp = Response(json.dumps(predictions))
    resp.headers['Content-type'] = 'application/json'
    return resp


if __name__ == '__main__':
    port = 12345
    # ip = socket.gethostbyname(socket.gethostname())
    # # app.run(host='0.0.0.0', port=port)
    # app.run(host=ip, port=port, debug=True)

    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()
