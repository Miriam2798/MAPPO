from flask import Flask
import json

app = Flask(__name__)
PORT = 5000
DEBUG = False

@app.errorhandler(404)
def not_found(error):
    return "Not Found."

@app.route('/', methods =['GET'])
def index():
    a = 2
    b = 10
    y = a + b
    
    
    #convertir en cadena JSON
    #jsonStr = json.dumps(str(y).__dict__)
    
    #imprimir la cadena json
    return str(y)

@app.route('/fastestroute')
def route():
    return "we are on fastest route tab"


if __name__=='__main__':
    app.run(port = PORT, debug = DEBUG)
