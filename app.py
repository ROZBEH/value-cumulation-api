from flask import Flask, request, abort, jsonify
import os
from chat_finance.chat_finance import index_sec_url

app = Flask(__name__)


@app.route("/")
def index():
    ############################## Athenticate the user ##############################
    request_api_key = request.headers.get("VALUE_CUMULATION_API_KEY")
    true_api_key = os.environ.get("API_KEY")
    if request_api_key != true_api_key:
        abort(401)  # Unauthorized
    ##################################################################################
    return "Hello World, you are authorized!"


@app.route("/greetings")
def greeting():
    ############################## Athenticate the user ##############################
    request_api_key = request.headers.get("VALUE_CUMULATION_API_KEY")
    true_api_key = os.environ.get("API_KEY")
    if request_api_key != true_api_key:
        abort(401)  # Unauthorized
    ##################################################################################

    return "Hello there and welcome!"


@app.route("/answer_financial_queries")
def answer_financial_queries():
    ############################## Athenticate the user ##############################
    request_api_key = request.headers.get("VALUE_CUMULATION_API_KEY")
    true_api_key = os.environ.get("API_KEY")
    if request_api_key != true_api_key:
        abort(401)
    ##################################################################################
    query = request.args.get("query")
    ticker = request.args.get("ticker")
    index = index_sec_url(ticker=ticker)
    query_engine = index.as_query_engine()
    results = query_engine.query(query)

    return jsonify(results)
