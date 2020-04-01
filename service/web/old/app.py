#!/bin/bash

import os.path
from os import path
import pickle

from flask import Flask, request
from flask_restful import reqparse, abort, Resource, Api

app = Flask(__name__)
api = Api(app)


cases = {}

cases_filename = 'cases.pickle'


welcome_message = """
Welcome to the DocSimilarity Service!

To start, you can try CURLing:

curl http://localhost:5000/cases # get all cases

"""


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



def abort_if_doesnt_exist(collection, entity_id):
    if entity_id not in collection:
        print("doesn't exist" + entity_id)
        print(collection)
        abort(404, message="Item {} doesn't exist".format(entity_id))

parser =  reqparse.RequestParser()
parser.add_argument('case')




class HelloWorld(Resource):
    def get(self):
        return {'smokingun': 'service'}

class Case(Resource):
    def get(self, case_id):
        abort_if_doesnt_exist(cases, case_id)
        return {case_id: cases[case_id]}

    def delete(self, case_id):
        abort_if_doesnt_exist(cases, case_id)
        del cases[case_id]
        pickle_files()
        return '', 204

    def put(self, case_id):
        args = parser.parse_args()
        print("args = " + str(args))
        new_entity = {'uri': args['uri']}
        cases[case_id] = new_entity
        pickle_files()
        return new_entity, 201

# CaseList
# shows a list of all cases, and lets you POST to add new cases
class CaseList(Resource):
    def get(self):
        return cases

    def post(self):
        args = parser.parse_args()
        print(args)
        case_id = args['case_id']
        cases[case_id] = {'uri': args['uri']}
        pickle_files()
        return cases[case_id], 201



unpickle_files()  # Load pickled files 


api.add_resource(HelloWorld, '/')
api.add_resource(Case, '/cases/<string:case_id>')
api.add_resource(CaseList, '/cases')

print(welcome_message)

if __name__ == '__main__':
    app.run(debug=True)

