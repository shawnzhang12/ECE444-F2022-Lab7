# -*- coding: utf-8 -*-
import flask
from flask import json

import os

import pickle

##############################################################################################
#
#   INIT
#
##############################################################################################

### model loading ###
model_path = "./"
loaded_model = None
with open(os.path.join(model_path,'basic_classifier.pkl'), 'rb') as fid:
    loaded_model = pickle.load(fid)

vectorizer = None
with open(os.path.join(model_path,'count_vectorizer.pkl'), 'rb') as vd:
    vectorizer = pickle.load(vd)

##############################################################################################
#
#   FLASK APP
#
##############################################################################################

# The flask app for serving predictions
application = flask.Flask(__name__)


@application.route('/')
def hello():
    return "Welcome to your own Sentiment Analysis Tool"

@application.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""

    status = 200 if (loaded_model and vectorizer)  else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@application.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single sentence.
    """
    data = None

    if flask.request.content_type == 'application/json':
        data= flask.request.get_json()

    else:
        return flask.Response(response='This predictor only supports Json data', status=415, mimetype='text/plain')

    text = data.get('text')
    prediction = loaded_model.predict(vectorizer.transform([text]))[0]

    result = {'text': text,
            'prediction': prediction
    }       

    return flask.Response(response=json.dumps(result), status=200, mimetype='application/json')



if __name__ == '__main__':
    application.run()
