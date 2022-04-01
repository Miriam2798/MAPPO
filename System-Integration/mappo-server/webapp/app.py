# Import OS to get the port environment variable from the Procfile
import os # <-----
import os.path
import taxicab as tc
import osmnx as ox
import networkx as nx
import time
import json
import requests
# Import the flask module
from flask import Flask

# Create a Flask constructor. It takes name of the current module as the argument
app = Flask(__name__)


@app.route('/')
def importFile():
    place = "Barcelona"
    if os.path.exists(f'{place}_graph.txt'):
        o = ox.load_graphml(f'{place}_graph.txt')
        return str(o)
    #print("\nFirst time running the script for " + place +
        #  ". Loading and Saving graph...\n")
    G = ox.graph_from_place(place, network_type='drive')
    ox.save_graphml(G, f"{place}_graph.txt")
    return str(G)

