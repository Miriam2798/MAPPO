# Import OS to get the port environment variable from the Procfile
import os # <-----
import json
import requests
import osmnx as ox
import networkx as nx
import taxicab as tc
import time
import os.path
import numpy as np
# Import the flask module
from flask import Flask, request
import fastestpath as fp
import ranking as rk

# Create a Flask constructor. It takes name of the current module as the argument
app = Flask(__name__)

@app.route('/fastestroute', methods = ['GET'])

def server_fastest_route():
    place = "Terrassa"
    originx = request.args.get("originx")
    originy = request.args.get("originy")
    destinationx = request.args.get("destinationx")
    destinationy = request.args.get("destinationy")

    ruta = fp.server(originx, originy, destinationx, destinationy, place)
    return ruta
    
@app.route('/rank', methods = ['GET'])

def server_rank():
    return str(rk.rank())

