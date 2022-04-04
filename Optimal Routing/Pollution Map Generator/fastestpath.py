import osmnx as ox
import networkx as nx
import taxicab as tc
import time
import os.path
import numpy as np
import csv


#Definition of a node object, with its attributes Node(float y, float x, int id)
class Node:

    def __init__(self, y, x, id):
        self.y = y
        self.x = x
        self.id = id

    # ToString function. Returns the id, and y, x coordinates
    def __repr__(self):
        return f"{self.id}, {self.y}, {self.x}"


# If exists, imports the graph from the file. The file is created after searching for a place once and for each one
def importFile(place):
    if os.path.exists(f'{place}_graph.txt'):
        return ox.load_graphml(f'{place}_graph.txt')
    print("\nFirst time running the script for " + place +
          ". Loading and Saving graph...\n")
    G = ox.graph_from_place(place, network_type='drive')
    ox.save_graphml(G, f"{place}_graph.txt")
    return G


# Calculates the fastest route of a place
def fastest_route(originx, originy, destinationx, destinationy, place):
    #Imports the graph
    G = importFile(place)
    #Generates tuples from the given coordinates
    origin_xy = tuple((float(originx), float(originy)))
    destination_xy = tuple((float(destinationx), float(destinationy)))
    #Finds the nearest node from a location point
    origin_node = ox.get_nearest_node(G, origin_xy)
    destination_node = ox.get_nearest_node(G, destination_xy)
    #Finds the shortest path and stores it into the route object. Using length argument to find the shortest path in terms of length
    route = nx.shortest_path(G=G,
                             source=origin_node,
                             target=destination_node,
                             weight='length')
    #Using Taxicab library to normalize the start and end of the route, as it won't always start in a node location
    routeTC = tc.distance.shortest_path(G, origin_xy, destination_xy)
    return routeTC, G, route


#Exports the route into a route.csv file
def export(G, routeTC):
    nodelist = []
    #Iterate the nodes to extrat all the coordinates along with its ids
    for i in range(int(len(routeTC[1]))):
        y = G.nodes[routeTC[1][i]]['y']
        x = G.nodes[routeTC[1][i]]['x']
        nodelist.append(Node(y, x, routeTC[1][i]))
    #Create and write the node stored into nodelist to a route.csv file
    with open("route.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in nodelist:
            writer.writerow([val])


def main():
    print(
        r"""___________                __                   __    __________         __  .__     
\_   _____/____    _______/  |_  ____   _______/  |_  \______   \_____ _/  |_|  |__  
 |    __) \__  \  /  ___/\   __\/ __ \ /  ___/\   __\  |     ___/\__  \\   __\  |  \ 
 |     \   / __ \_\___ \  |  | \  ___/ \___ \  |  |    |    |     / __ \|  | |   Y  \
 \___  /  (____  /____  > |__|  \___  >____  > |__|    |____|    (____  /__| |___|  /
     \/        \/     \/            \/     \/                         \/          \/"""
    )
    time.sleep(1)
    place = input("Please, insert the place name: (example: Barcelona) \n")
    originx, originy = input(
        "Please, insert the origin coordinates: (example: 41.59047, 2.45235) \n"
    ).split(", ")
    destinationx, destinationy = input(
        "Please, insert the destination coordinates: (example: 41.59047, 2.45235) \n"
    ).split(", ")
    #Calculate fastest route given the inputs
    routeTC, G, route = fastest_route(originx, originy, destinationx,
                                      destinationy, place)
    #Export the route into a route.csv file
    export(G, routeTC)

    #Plo the route in red color
    fig, ax = tc.plot.plot_graph_route(
        G,
        routeTC,
        route_color="r",
        orig_dest_size=100,
        ax=None,
    )