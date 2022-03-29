# Import OS to get the port environment variable from the Procfile
import os # <-----
# Import the flask module
from flask import Flask

# Create a Flask constructor. It takes name of the current module as the argument
app = Flask(__name__)

@app.route('/')
def hello_world():
    statement = 'Hello World!'
    return statement

