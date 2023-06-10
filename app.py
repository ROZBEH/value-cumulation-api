import os
import logging
from logger import custom_logger
from flask_cors import CORS

from flask import Flask, request, abort, jsonify
from chat_finance.chat_finance import index_sec_url, chat_bot_agent

app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:8910"}})


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


# @app.route("/answer_financial_queries")
# def answer_financial_queries():
#     print("answer financials is called...")
#     ############################## Athenticate the user ##############################
#     request_api_key = request.headers.get("VALUE_CUMULATION_API_KEY")
#     true_api_key = os.environ.get("API_KEY")
#     if request_api_key != true_api_key:
#         abort(401)
#     ##################################################################################
#     query = request.args.get("query")
#     ticker = request.args.get("ticker")
#     index = index_sec_url(ticker=ticker)
#     query_engine = index.as_query_engine()
#     results = query_engine.query(query)
#     return jsonify(results)


@app.route("/answer_financial_queries", methods=["POST"])
def answer_financial_queries():
    # Get user input from POST request
    text_input = request.json.get("message")

    # Create chat_bot_agent and get response
    agent_chain = chat_bot_agent(load_index=True)
    response = agent_chain.run(input=text_input)

    return jsonify({"response": response})


@app.route("/logs")
def get_logs():
    from logger import handler

    logs = handler.buffer.copy()  # Copy the buffer
    # remove the following line if you don't want to clear the logs
    handler.clear()  # Clear the buffer

    return jsonify({"logs": logs})
