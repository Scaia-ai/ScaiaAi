from flask import Flask, jsonify, request, abort
from os import path,mkdir,chdir,path
import pickle
import wget
import os
import zipfile


app = Flask(__name__)



cases_filename = 'cases.pickle'


welcome_message = """
Welcome to the DocSimilarity Service!

To start, you can try CURLing:

curl http://localhost:5000/cases # get all cases
curl http://localhost:5000/case/1 # get all cases

curl -X POST -H "Content-Type: application/json" -d '{
  "case_id": "1",
  "uri": "my_uri"
}' http://localhost:5000/cases



"""

cases={}
case_files={}

def pickle_files():
    outfile = open(cases_filename, 'wb')
    pickle.dump(cases, outfile)
    outfile.close()

def unpickle_files():
    global cases
    if path.exists(cases_filename):
        infile = open(cases_filename, 'rb')
        cases = pickle.load(infile)
        infile.close()


unpickle_files()
print("Current state of cases: " + str(cases))


def get_files():
    files=[]
    for r,d,f in os.walk('.'):
        for t_file in f:
            if 'zip' not in t_file:
                files.append(os.path.join(r, t_file))
    return(files)

def download_uri(case_id, uri):
    if not os.path.exists(case_id):
       mkdir(case_id)
    chdir(case_id)
    print(uri)
    filename = wget.download(uri.strip())
    print(filename)
    if '.zip' in filename:
       with zipfile.ZipFile(filename, 'r') as zip_ref:
          zip_ref.extractall('.')
    t_files = get_files()
    case_files[case_id] = t_files
    cases[case_id]['files'] = t_files 
    chdir("..")
    print(case_files)
    pickle_files()


@app.route("/")
def hello_world():
  return "Hello, World!"


@app.route('/case/<int:case_id>', methods=['GET'])
def get_case(case_id):
    case_id_str = str(case_id)
    print("getting case_id: " + case_id_str)
    case = cases[case_id_str] if case_id_str in cases else None 

    if not case:
        case = case[str(case_id)] if case_id in cases else None
        if not case: 
            abort(404)
    return jsonify({'case': case})


@app.route('/cases')
def get_cases():
  return jsonify(cases)


@app.route('/cases', methods=['POST'])
def add_case():
  this_case = request.get_json()
  print(this_case)
  if this_case:  # not null
     cases[this_case['case_id']] = this_case
     pickle_files()
     download_uri(this_case['case_id'], this_case['uri'])
  return '', 204
