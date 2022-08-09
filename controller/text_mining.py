import jsonschema

from engine.test import Testing

from rules.text_mining_rules import *
from flask import Response, request, json
from resources.validations.error_messages import *


def index():
    response_text = '{ "message": "Hello, welcome to views predictions api" }'
    response = Response(response_text, 200, mimetype='application/json')
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


def find():
    request_data = request.get_json()
    try:
        testing = Testing()
        jsonschema.validate(request_data, text_mining_find_schema)
        clean_title = testing.title_cleaner([request_data['title']])
        predict_result = testing.predict(clean_title)
        response = Response(json.dumps(str(predict_result)), 201, mimetype="application/json")
    except jsonschema.exceptions.ValidationError as exc:
        response = Response(error_message_helper(exc.message), 400, mimetype="application/json")
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


def confusion_get():
    try:
        testing = Testing()
        results = testing.confusion_matrix()

        response = Response(json.dumps(str(results)), 201, mimetype="application/json")
    except jsonschema.exceptions.ValidationError as exc:
        response = Response(error_message_helper(exc.message), 400, mimetype="application/json")
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


def classification_get():
    try:
        testing = Testing()
        results = testing.classification_report()

        response = Response(json.dumps(str(results)), 201, mimetype="application/json")
    except jsonschema.exceptions.ValidationError as exc:
        response = Response(error_message_helper(exc.message), 400, mimetype="application/json")
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response
