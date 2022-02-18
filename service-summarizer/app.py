from flask import Flask, jsonify, request, abort
from os import path,mkdir,chdir,path
import pickle
import wget
import os
import zipfile

import hf_pegasus


app = Flask(__name__)




welcome_message = """
Welcome to the Summarizer Service!

To start, you can try CURLing:

curl -X POST -H "Content-Type: application/json" -d '{
  "text": "This is the document to summarize"
}' http://localhost:5000/summarizeText



"""


@app.route("/")
def hello_world():
  return welcome_message

@app.route('/summarizeTextTest', methods=['POST'])
def summarize_text():
  if not request.json or not 'text' in request.json:
      abort(400)
  this_document = request.get_json()
  if this_document:  # not null
     text_to_summarize = this_document['text'] 
     doc = {
       'summary': text_to_summarize
     }
     return jsonify(doc)
  return '', 204

@app.route('/summarizeText', methods=['POST'])
def summarize_text_pg():
  if not request.json or not 'text' in request.json:
      abort(400)
  this_document = request.get_json()
  if this_document:  # not null
     text_to_summarize = this_document['text'] 
     print(text_to_summarize)
     summarized_text = hf_pegasus.summarize(text_to_summarize)
     doc = {
       'summary': summarized_text
     }
     return jsonify(doc)
  return '', 204

@app.route('/summarizeTextModel', methods=['POST'])
def summarize_text_model():
  if not request.json or not 'text' in request.json:
      abort(400)
  this_document = request.get_json()
  if this_document:  # not null
     text_to_summarize = this_document['text'] 
     model_name = this_document['text'] 
     print(text_to_summarize)
     summarized_text = hf_pegasus.summarize_model(text_to_summarize, model_name)
     doc = {
       'summary': summarized_text
     }
     return jsonify(doc)
  return '', 204

@app.route('/summarizeTextLegal', methods=['POST'])
def summarize_text_legal():
  if not request.json or not 'text' in request.json:
      abort(400)
  this_document = request.get_json()
  if this_document:  # not null
     text_to_summarize = this_document['text'] 
     print(text_to_summarize)
     summarized_text = hf_pegasus.summarize_legal(text_to_summarize)
     doc = {
       'summary': summarized_text
     }
     return jsonify(doc)
  return '', 204


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
