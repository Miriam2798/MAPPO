#Pollution Map Generator
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import osmnx as ox
import networkx as nx
import taxicab as tc
import os.path
import csv
import warnings
from pandas.core.common import SettingWithCopyWarning
import pandas as pd
import folium
from folium.plugins import HeatMap
import taxicab as tc
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool


#Remove warnings
def removeWarnings():
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    warnings.simplefilter("ignore", UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


# Heatmap Plotting function
def data_coord2view_coord(p, vlen, pmin, pmax):
    dp = pmax - pmin
    return (p - pmin) / dp * vlen


# Nearest Neighbour
def nearest_neighbours(xs, ys, reso, n_neighbours):
    im = np.zeros([reso, reso])
    extent = [np.min(xs), np.max(xs), np.min(ys), np.max(ys)]
    xv = data_coord2view_coord(xs, reso, extent[0], extent[1])
    yv = data_coord2view_coord(ys, reso, extent[2], extent[3])
    for x, y in itertools.product(range(reso), range(reso)):
        xp = (xv - x)
        yp = (yv - y)
        d = np.sqrt(xp**2 + yp**2)
        im[y][x] = 1 / np.sum(d[np.argpartition(d.ravel(),
                                                n_neighbours)[:n_neighbours]])
    return im, extent


# Export to csv file called map.csv
def exportMap(im):
    np.savetxt("map.csv", im, delimiter=",")


#Pollution Map Generator function
def pollutionMapGenerator(generated_points, resolution, nn):
    removeWarnings()
    # Generating random normal values from (0,1)
    x = np.random.randn(int(generated_points))
    x = x - min(x)
    y = np.random.randn(int(generated_points))
    y = y - min(y)
    # Store the values into im
    im, extent = nearest_neighbours(x, y, int(resolution), int(nn))
    # Normalize the values to (0,1)
    im = (im - np.min(im)) / np.ptp(im)
    # Plot im as a heatmap
    plt.imshow(im, origin='lower', extent=extent, cmap=cm.turbo)
    plt.show()
    # Export as a csv file
    exportMap(im)


def mainPollutionMapGenerator():
    print(
        r"""__________      .__  .__          __  .__                   _____                 
\______   \____ |  | |  |  __ ___/  |_|__| ____   ____     /     \ _____  ______  
 |     ___/  _ \|  | |  | |  |  \   __\  |/  _ \ /    \   /  \ /  \\__  \ \____ \ 
 |    |  (  <_> )  |_|  |_|  |  /|  | |  (  <_> )   |  \ /    Y    \/ __ \|  |_> >
 |____|   \____/|____/____/____/ |__| |__|\____/|___|  / \____|__  (____  /   __/ 
                                                     \/          \/     \/|__|"""
    )
    time.sleep(1)
    generated_points = input(
        "Please specify the number of points generated (100 to 10000) default=1000: \n"
    )
    resolution = input(
        "Please specify the resolution (250 to 500) default=100: \n")
    nn = input(
        "Please, specify the Nearest Neighbour parameter (default=16): \n")
    return generated_points, resolution, nn


#Fastest Route Computing


#Definition of a node object, with its attributes Node(float y, float x, int id)
class Node:

    def __init__(self, y, x, id, value):  #, value
        self.y = y
        self.x = x
        self.id = id
        self.value = value

    # ToString function. Returns the id, and y, x coordinates
    def __repr__(self):
        return f"{self.id}, {self.y}, {self.x},{self.value}"


# If exists, imports the graph from the file. The file is created after searching for a place once and for each one
def importFile(place):
    if os.path.exists(f'{place}_graph.txt'):
        return ox.load_graphml(f'{place}_graph.txt')
    print("\nFirst time running the script for " + place +
          ". Loading and Saving graph...\n")
    G = ox.graph_from_place(place, network_type='drive')
    ox.save_graphml(G, f"{place}_graph.txt")
    return G


def mainFastestRoute():
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
    return place, tuple((originx, originy)), tuple(
        (destinationx, destinationy))


# Calculates the fastest route of a place
def fastest_route(originx, originy, destinationx, destinationy, place):
    removeWarnings()
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
    filename = "fastestroute.csv"
    #fig, ax = tc.plot.plot_graph_route(
    #    G,
    #    routeTC,
    #    route_color="r",
    #    orig_dest_size=100,
    #    ax=None,
    #)
    return exportTC(G, route, filename)


#Exports the route into a route.csv file
def export(G, routeTC, filename):
    nodelist = []
    #Iterate the nodes to extrat all the coordinates along with its ids
    for i in range(int(len(routeTC))):
        y = G.nodes[routeTC[i]]['y']
        x = G.nodes[routeTC[i]]['x']
        value = G.nodes[routeTC[i]]['Pollution']
        nodelist.append(Node(y, x, routeTC[i], value))
    #Create and write the node stored into nodelist to a route.csv file
    #with open(filename, "w") as output:
    #    writer = csv.writer(output, lineterminator='\n')
    #    for val in nodelist:
    #        writer.writerow([val])
    return str(nodelist)


def exportTC(G, routeTC, filename):
    nodelist = []
    #Iterate the nodes to extrat all the coordinates along with its ids
    for i in range(int(len(routeTC[1]))):
        y = G.nodes[routeTC[1][i]]['y']
        x = G.nodes[routeTC[1][i]]['x']
        nodelist.append(Node(y, x, routeTC[1][i]))
    #Create and write the node stored into nodelist to a route.csv file
    #with open(filename, "w") as output:
    #    writer = csv.writer(output, lineterminator='\n')
    #    for val in nodelist:
    #        writer.writerow([val])
    return str(nodelist)


#Fastest Route function
def fastestRoute(place, originx, originy, destinationx, destinationy):
    removeWarnings()
    #Calculate fastest route given the inputs
    routeTC, G, route = fastest_route(originx, originy, destinationx,
                                      destinationy, place)
    #Plot the route in red color
    #fig, ax = tc.plot.plot_graph_route(
    #    G,
    #    routeTC,
    #    route_color="r",
    #    orig_dest_size=100,
    #    ax=None,
    #)
    filename = "fastestroute.csv"
    return exportTC(G, routeTC, filename)


#Less Polluted Route
#Defines a Point object with its own atributes Point(float y, float x, float PollutionValue,int node, tuple edge)
class Point:

    def __init__(self, y, x, value):
        self.y = y
        self.x = x
        self.value = value

    #ToString function
    def __repr__(self):
        return f"[{self.y}, {self.x}, {self.value}]"

    #Getters
    def getY(self):
        return self.y

    def getX(self):
        return self.x

    def getValue(self):
        return self.value

    def getNode(self):
        return self.node

    def getEdge(self):
        return self.edge

    def getNdist(self):
        return self.ndist

    def getEdist(self):
        return self.edist

    #Setters
    def setNode(self, node):
        self.node = node

    def setEdge(self, edge):
        self.edge = edge

    def setNdist(self, ndist):
        self.ndist = ndist

    def setEdist(self, edist):
        self.edist = edist


def dataMapping(origin_yx, destination_yx, city, reso, increment):
    removeWarnings()
    G = importFile(city)
    Gnx = nx.relabel.convert_node_labels_to_integers(G)
    nodes, edges = ox.graph_to_gdfs(Gnx, nodes=True, edges=True)
    nodes['Pollution'] = float(0)
    pollutionMatrix = np.loadtxt(open("map.csv", "rb"),
                                 delimiter=",",
                                 skiprows=1)
    rows = float(round(max(nodes['y']) - min(nodes['y']), increment))
    cols = float(round(max(nodes['x']) - min(nodes['x']), increment))
    incrow = round(rows / reso, increment)
    incol = round(cols / reso, increment)
    points = []
    y = max(nodes['y'])
    x = min(nodes['x'])
    #Iteration that gives geolocation values to every value in the pollution matrix. Objects Point(y,x,value) are stored into a points list
    for row in range(len(pollutionMatrix)):
        for col in range(len(pollutionMatrix[row])):
            if x > max(nodes['x']):
                x = round(min(nodes['x']), 4)
            points.append(
                Point(round(y, 4), round(x, 4), pollutionMatrix[row][col]))
            x = x + incol
        if y <= round(min(nodes['y']), 4) and x >= round(max(nodes['x']), 4):
            break
        else:
            y = y - incrow
    return points


#Set Pollution values to edges
def set_values_to_edges(points, edges, G):
    edist = 0
    first = True
    for p in range(len(points)):
        y = points[p].getY()
        x = points[p].getX()
        point = tuple((y, x))
        ne, edist = ox.distance.nearest_edges(G, x, y, return_dist=True)
        if first:
            epdist = edist
            first = False
        if edist <= epdist:
            print("Edge " + str(ne[0]) + " " + str(ne[1]) + " " + str(ne[2]) +
                  " has a value of " + str(points[p].getValue()) +
                  " and the nearest point is " + str(point) +
                  " at a distance of " + str(edist) + "\n")
            G[ne[0]][ne[1]][0]['Pollution'] = 1 - points[p].getValue()
        points[p].setEdge((ne[0], ne[1]))
        points[p].setEdist(edist)
    return edges


#Set pollution values to nodes
def set_values_to_nodes(points, nodes, G):
    pdist = 0
    first = True
    nodes['Pollution'] = float(0)
    for p in range(len(points)):
        y = points[p].getY()
        x = points[p].getX()
        point = tuple((y, x))
        selectedNode, dist = ox.distance.nearest_nodes(G,
                                                       x,
                                                       y,
                                                       return_dist=True)
        if first:
            pdist = dist
            first = False
        if dist <= pdist:
            print("Node " + str(selectedNode) + " has a value of " +
                  str(points[p].getValue()) + " and the nearest point is " +
                  str(point) + " at a distance of " + str(dist))
            nodes['Pollution'][selectedNode] = 1 - points[p].getValue()
        points[p].setNode(selectedNode)
        points[p].setNdist(dist)
    return nodes


#Export map as route.html using folium
def mapFolium(G2, route, filepath, originyx, destinationyx, city):
    d = pd.read_csv('points_' + city + '.csv')
    df = pd.DataFrame(d)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.loc[:, ~df.columns.str.contains('node')]
    df = df.loc[:, ~df.columns.str.contains('edge')]
    df = df.loc[:, ~df.columns.str.contains('ndist')]
    df = df.loc[:, ~df.columns.str.contains('edist')]

    
    HeatMap(df,
            radius=10,
            max_zoom=10,
            gradient={
                0.05: 'blue',
                0.1: 'lime',
                0.3: 'yellow',
                0.4: 'orange',
                0.6: 'red'
            }).add_to(route_map)
    folium.Marker(location=[originyx[0], originyx[1]],
                  popup='Origen').add_to(route_map)
    folium.Marker(location=[destinationyx[0], destinationyx[1]],
                  popup='Destino').add_to(route_map)
    if filepath == "":
        filepath = 'LessPollutedRoute.html'
    route_map.save(filepath)


def mainLessPollutedRoute():
    print(
        r""".____                          __________      .__  .__          __             .___ __________               __          
|    |    ____   ______ ______ \______   \____ |  | |  |  __ ___/  |_  ____   __| _/ \______   \ ____  __ ___/  |_  ____  
|    |  _/ __ \ /  ___//  ___/  |     ___/  _ \|  | |  | |  |  \   __\/ __ \ / __ |   |       _//  _ \|  |  \   __\/ __ \ 
|    |__\  ___/ \___ \ \___ \   |    |  (  <_> )  |_|  |_|  |  /|  | \  ___// /_/ |   |    |   (  <_> )  |  /|  | \  ___/ 
|_______ \___  >____  >____  >  |____|   \____/|____/____/____/ |__|  \___  >____ |   |____|_  /\____/|____/ |__|  \___  >
        \/   \/     \/     \/                                             \/     \/          \/                        \/ """
    )
    time.sleep(1)
    city = input("Please, insert the place name: (example: Barcelona) \n")
    originy, originx = input(
        "Please, insert the origin coordinates: (example: 41.59047, 2.45235) \n"
    ).split(", ")
    destinationy, destinationx = input(
        "Please, insert the destination coordinates: (example: 41.59047, 2.45235) \n"
    ).split(", ")
    G = importFile(city)
    #Gnx = nx.relabel.convert_node_labels_to_integers(G)
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    return city, tuple((originy, originx)), tuple(
        (destinationy, destinationx)), nodes, edges, G


#Less Polluted route function
def LessPollutedRoute(originx, originy, destinationx, destinationy, city, reso,
                      increment, nodes, edges, G):
    removeWarnings()
    #Gnx = nx.relabel.convert_node_labels_to_integers(G)
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    if os.path.exists('points_' + city + '.csv'):
        d = pd.read_csv('points_' + city + '.csv')
        df = pd.DataFrame(d)
        #nodes
        df = df.sort_values(by=['ndist'])
        df = df.drop_duplicates(keep='first', subset='node')
        nodes['Pollution'] = float(0)
        for ind in df.index:
            nodes['Pollution'][int(df['node'][ind])] = 1 - df['value'][ind]
        #edges
        df = df.sort_values(by=['edist'])
        df = df.drop_duplicates(keep='first', subset='edge')
        for ind in df.index:
            id = df['edge'][ind].split(",")
            id[0] = int(id[0].replace("(", ""))
            id[1] = int(id[1].replace(")", ""))
            G[id[0]][id[1]][0]['Pollution'] = 1 - df['value'][ind]
        ripnodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True) # G is not used :(
        G2 = ox.graph_from_gdfs(nodes, edges)
    else:
        origin_yx = tuple((originy, originx))
        destination_yx = tuple((destinationy, destinationx))
        print("\nFirst time running the script. Mapping the data...\n")
        points = dataMapping(origin_yx, destination_yx, city, reso, increment)

        # cores = mp.cpu_count()
        # points_split = np.array_split(points, cores, axis=0)
        # print(points_split)
        # pool = Pool(cores)
        # nodes = np.vstack(
        # pool.map(set_values_to_nodes(points, nodes, Gnx), points_split))
        # pool.close()
        # pool.join()
        # pool.clear()

        tic = time.time()
        nodes = set_values_to_nodes(points, nodes, G)
        toc = time.time()
        print("Node processing lasted: " + str(toc - tic))
        tic = time.time()
        G = set_values_to_edges(points, edges, G)
        toc = time.time()
        print("Edge processing lasted: " + str(toc - tic))
        G2 = ox.graph_from_gdfs(nodes, edges)
        lat = []
        lon = []
        val = []
        node = []
        edge = []
        ndist = []
        edist = []
        for p in range(len(points)):
            lat.append(points[p].getY())
            lon.append(points[p].getX())
            val.append(points[p].getValue())
            node.append(points[p].getNode())
            edge.append(points[p].getEdge())
            ndist.append(points[p].getNdist())
            edist.append(points[p].getEdist())
        d = {
            'lat': lat,
            'lon': lon,
            'value': val,
            'node': node,
            'edge': edge,
            'ndist': ndist,
            'edist': edist
        }
        df = pd.DataFrame(d)
        df.to_csv('points_' + city + '.csv')
    origin_yx = tuple((float(originy), float(originx)))
    destination_yx = tuple((float(destinationy), float(destinationx)))
    origin_node = ox.get_nearest_node(G2, origin_yx)
    destination_node = ox.get_nearest_node(G2, destination_yx)
    route = nx.shortest_path(G=G2,
                             source=origin_node,
                             target=destination_node,
                             weight='Pollution')
    fastroute = nx.shortest_path(G=G2,
                                 source=origin_node,
                                 target=destination_node,
                                 weight='length')
    #routeTC = tc.distance.shortest_path(G, origin_yx, destination_yx)
    filepath = 'route.html'
    #mapFolium(G2, route, filepath, origin_yx, destination_yx, city)
    rc = ['r', 'g']
    #fig, ax = ox.plot_graph_routes(G2, [fastroute, route],
    #                               route_colors=rc,
    #                               route_linewidth=6,
    #                               node_size=0)
    # fig, ax = ox.plot_graph_route(
    #     G2,
    #     route,
    #     route_color="r",
    #     orig_dest_size=100,
    #     ax=None,
    # )
    filename = "lesspollutedroute.csv"
    
    return export(G2, route, filename), export(G2, fastroute, "fastestroute.csv")