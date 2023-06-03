from flask import Flask, request, abort
import os

app = Flask(__name__)


@app.route("/")
def index():
    request_api_key = request.headers.get("VALUE_CUMULATION_API_KEY")
    true_api_key = os.environ.get("API_KEY")
    if request_api_key != true_api_key:
        abort(401)  # Unauthorized
    return "Hello World, you are authorized!"
