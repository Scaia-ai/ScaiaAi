#!/bin/bash

import os.path
from os import path
import pickle
from pymongo import MongoClient

from flask import Flask, request
from flask_restful import reqparse, abort, Resource, Api

app = Flask(__name__)
api = Api(app)


users = {}
projects = {}
uris = {}



client = MongoClient("mongodb://my_db:27017")
db = client.projectDB

welcome_message = """
Welcome to the DocSimilarity Service!

To start, you can try CURLing:

curl http://localhost:5000/users # get all users
curl http://localhost:5000/projects # get all projects

"""


def pickle_files():

    outfile = open(users_filename, 'wb')
    db['Users'].insert(users)
    db['Projects'].insert(users)
    db['Uris'].insert(users)

def unpickle_files():
    global users, projects, uris
    if path.exists(users_filename):
        infile = open(users_filename, 'rb')
        users = pickle.load(infile)
        infile.close()
    if path.exists(projects_filename):
        infile = open(projects_filename, 'rb')
        projects = pickle.load(infile)
        infile.close()
    if path.exists(uris_filename):
        infile = open(uris_filename, 'rb')
        uris = pickle.load(infile)
        infile.close()





def abort_if_doesnt_exist(collection, entity_id):
    if entity_id not in collection:
        print("doesn't exist" + entity_id)
        print(collection)
        abort(404, message="Item {} doesn't exist".format(entity_id))

parser =  reqparse.RequestParser()
parser.add_argument('user')
parser.add_argument('project')
parser.add_argument('fileuri')




class HelloWorld(Resource):
    def get(self):
        return {'docsimilarity': 'service'}

class User(Resource):
    def get(self, user_id):
        abort_if_doesnt_exist(users, user_id)
        return users[user_id]

    def delete(self, user_id):
        abort_if_doesnt_exist(user_id)
        del users[user_id]
        pickle_files()
        return '', 204

    def put(self, user_id):
        args = parser.parse_args()
        new_entity = {'user': args['user']}
        users[user_id] = new_entity
        pickle_files()
        return new_entity, 201


# UserList
# shows a list of all users, and lets you POST to add new users
class UserList(Resource):
    def get(self):
        return users

    def post(self):
        args = parser.parse_args()
        user_id = int(max(users.keys()).lstrip('user')) + 1
        user_id = 'user%i' % user_id
        users[user_id] = {'user': args['user']}
        pickle_files()
        return users[user_id], 201



class Project(Resource):
    def get(self, project_id):
        abort_if_doesnt_exist(projects, project_id)
        return {project_id: projects[project_id]}

    def delete(self, project_id):
        abort_if_doesnt_exist(projects, project_id)
        del projects[project_id]
        pickle_files()
        return '', 204

    def put(self, project_id):
        args = parser.parse_args()
        new_entity = {'project': args['project'], 'user': args['user']}
        projects[project_id] = new_entity
        pickle_files()
        return new_entity, 201

# ProjectList
# shows a list of all projects, and lets you POST to add new projects
class ProjectList(Resource):
    def get(self):
        return projects

    def post(self):
        args = parser.parse_args()
        project_id = int(max(projects.keys()).lstrip('project')) + 1
        project_id = 'project%i' % project_id
        projects[project_id] = {'project': args['project'], 'user': args['user']}
        pickle_files()
        return projects[project_id], 201


class FileUri(Resource):
    def get(self, fileuri_id):
        abort_if_doesnt_exist(uris, fileuri_id)
        return uris[fileuri_id]

    def delete(self, fileuri_id):
        abort_if_doesnt_exist(uris, fileuri_id)
        del uris[fileuri_id]
        pickle_files()
        return '', 204

    def put(self, fileuri_id):
        args = parser.parse_args()
        new_entity = {'fileuri' : args['fileuri'], 'project': args['project']}
        uris[fileuri_id] = new_entity
        pickle_files()
        return new_entity, 201


# UserList
# shows a list of all users, and lets you POST to add new users
class FileUriList(Resource):
    def get(self):
        return uris

    def post(self):
        args = parser.parse_args()
        entity_id = int(max(users.keys()).lstrip('user')) + 1
        entity_id = 'uri%i' % entity_id
        uris[entity_id] = {'fileuri' : args['fileuri'], 'project': args['project']}
        pickle_files()
        return uris[entity_id], 201




unpickle_files()  # Load pickled files 


api.add_resource(HelloWorld, '/')
api.add_resource(User, '/users/<string:user_id>')
api.add_resource(UserList, '/users')
api.add_resource(Project, '/projects/<string:project_id>')
api.add_resource(ProjectList, '/projects')
api.add_resource(FileUri, '/fileuris/<string:fileuri_id>')
api.add_resource(FileUriList, '/fileuris')

print(welcome_message)

if __name__ == '__main__':
    app.run(debug=True)

