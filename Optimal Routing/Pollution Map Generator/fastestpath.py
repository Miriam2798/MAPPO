import osmnx as ox
import networkx as nx
import taxicab as tc
import time
import os.path


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
    #G = ox.graph_from_place(place, network_type='drive')
    G = importFile(place)
    origin_xy = tuple((float(originx), float(originy)))
    destination_xy = tuple((float(destinationx), float(destinationy)))
    origin_node = ox.get_nearest_node(G, origin_xy)
    destination_node = ox.get_nearest_node(G, destination_xy)
    route = nx.shortest_path(G=G,
                             source=origin_node,
                             target=destination_node,
                             weight='length')
    routeTC = tc.distance.shortest_path(G, origin_xy, destination_xy)
    return routeTC, G


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
    routeTC, G = fastest_route(originx, originy, destinationx, destinationy,
                               place)
    fig, ax = tc.plot.plot_graph_route(G, routeTC)