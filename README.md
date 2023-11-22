# value-cumulation-api

To run your Flask application `app.py`, you will need to follow a few steps:

### Set Environment Variables:

You need to set the environment variables `FLASK_ENV` and `API_KEY``. You need to set these variables in your environment before running the app.
On a Unix-like system, you can set them temporarily in your terminal session like this:

```
$ export FLASK_ENV=development
$ export API_KEY=your_api_key
```

Replace `your_api_key` with the actual API key you intend to use.
On Windows, use set instead of export:

```
$ set FLASK_ENV=development
$ set API_KEY=your_api_key
```

### Install the requirements

Run `pip install -r requirements.txt` in order to install requirements needed for the application. Please make sure to create a virtual environment dedicated for this purpose.

### Run the Flask Application

Navigate to the directory containing your `app.py` file. Run the application using:

```
$ python app.py
```

This command will start a development server for your Flask application.

### Access the Application:

Once the server is running, you can access your Flask application by opening a web browser and navigating to `http://127.0.0.1:5000/`

To test other routes like `/answer_financial_queries` or `/greetings`, you can use tools like Postman or cURL, ensuring you include the correct VALUE_CUMULATION_API_KEY in the request headers. Or you can directly call this on the client side.

### Logging Information:

Our application also includes custom logging in order to give you a better sense of what's
happening in the backend.

### Debug Mode

Running Flask in debug mode (`FLASK_ENV=development`) is beneficial for development since it provides a reloader and debugger. However, it should never be used in a production environment due to security concerns.
