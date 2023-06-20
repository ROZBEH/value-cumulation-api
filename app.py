import os
import logging
from logger import custom_logger
from flask import Flask, request, abort, jsonify
from flask_cors import CORS
from chat_finance.chat_finance import chat_bot_agent


def create_app():
    app = Flask(__name__)
    CORS(app)

    # Set configuration variables
    app.config.from_mapping(
        DEBUG=os.environ.get("FLASK_ENV") == "development",
        # other configs
    )

    with app.app_context():
        app.agent_chain = chat_bot_agent()

    @app.route("/answer_financial_queries", methods=["POST"])
    def answer_financial_queries():
        ############################## Athenticate the user ##############################
        request_api_key = request.headers.get("VALUE_CUMULATION_API_KEY")
        true_api_key = os.environ.get("API_KEY")
        if request_api_key != true_api_key:
            abort(401)
        ##################################################################################
        # Get user input from POST request
        text_input = request.json.get("message")

        # Get response from the preloaded agent_chain
        response = app.agent_chain.run(input=text_input)

        return jsonify({"response": response})

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

        return app

    @app.route("/logs")
    def get_logs():
        from logger import handler

        logs = handler.buffer.copy()  # Copy the buffer
        # remove the following line if you don't want to clear the logs
        handler.clear()  # Clear the buffer

        return jsonify({"logs": logs})

    return app


if __name__ == "__main__":
    custom_logger.info("Starting Flask app...")
    app = create_app()
    app.run(debug=app.config["DEBUG"])
    custom_logger.info("APP created...")
