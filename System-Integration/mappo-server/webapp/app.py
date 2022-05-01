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
import mappoAPI as api
import database as db

# Create a Flask constructor. It takes name of the current module as the argument
app = Flask(__name__)
    
@app.route('/database', methods = ['GET'])

def server_database():

    option = request.args.get("option") #options are: rank, rankthismonth, contno2, userdata
    return str(db.database(option))


@app.route('/routing', methods = ['GET'])

def server_less_polluted_route():

    option = request.args.get("option") #options are fastest, lesspolluted
    place = request.args.get("place")
    originx = request.args.get("originx")
    originy = request.args.get("originy")
    destinationx = request.args.get("destinationx")
    destinationy = request.args.get("destinationy")
    G = api.importFile(place)
    #Gnx = nx.relabel.convert_node_labels_to_integers(G)
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    lesspolluted, fastest = api.LessPollutedRoute(originy, originx, destinationy, 
                            destinationx, place, 100, 4, nodes,
                            edges, G)

    rutes = [lesspolluted, fastest]

    if option == "fastest":
        return rutes[1]
    elif option == "lesspolluted":
        return rutes[0]

