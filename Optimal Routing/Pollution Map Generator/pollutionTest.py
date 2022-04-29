# TEST SCRIPT, DO NOT TRY TO USE LOL
import warnings
from pandas.core.common import SettingWithCopyWarning

import osmnx as ox
import networkx as nx
import numpy as np
import math
import fastestpath as fp
import pandas as pd
import folium
from folium.plugins import HeatMap
import time

#Remove warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


#Defines a Point object with its own atributes Point(float y, float x, float PollutionValue)
class Point:

    def __init__(self, y, x, value):
        self.y = y
        self.x = x
        self.value = value

    #ToString function
    def __repr__(self):
        return f"[{self.y}, {self.x}, {self.value},{self.node},{self.edge},{self.ndist}, {self.edist}]"

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


city = input("Which city do you want to use? (Example: Barcelona) \n")
origin_yx = input("Origin point: \n")
destination_yx = input("Destination point: \n")
if (city == ""):
    city = "Terrassa"
if origin_yx == "" and destination_yx == "":
    origin_yx = (41.56538214481594, 2.010169694600521)
    destination_yx = (41.56312304751537, 2.0302406158505706)
G = fp.importFile(city)
Gnx = nx.relabel.convert_node_labels_to_integers(G)
nodes, edges = ox.graph_to_gdfs(Gnx, nodes=True, edges=True)

nodes['Pollution'] = float(0)

pollutionMatrix = np.loadtxt(open("map.csv", "rb"), delimiter=",", skiprows=1)
rows = float(round(max(nodes['y']) - min(nodes['y']), 4))
cols = float(round(max(nodes['x']) - min(nodes['x']), 4))

reso = 10
incrow = round(rows / reso, 4)
incol = round(cols / reso, 4)

# print(max(nodes['x']))
# print(min(nodes['x']))
# print(max(nodes['y']))
# print(min(nodes['y']))
# print(float(incrow), float(incol))
matrix = np.zeros((int(rows), int(cols)))  # np.zeros(int(rows),int(cols))

# Having troubles running through the matrix
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


#print(points)
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
    return G


def set_values_to_nodes(points, nodes, Gnx):
    pdist = 0
    first = True
    for p in range(len(points)):
        y = points[p].getY()
        x = points[p].getX()
        point = tuple((y, x))
        selectedNode, dist = ox.distance.nearest_nodes(Gnx,
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


def mapFolium(G2, route):
    d = pd.read_csv('points.csv')
    df = pd.DataFrame(d)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.loc[:, ~df.columns.str.contains('node')]
    df = df.loc[:, ~df.columns.str.contains('edge')]
    df = df.loc[:, ~df.columns.str.contains('ndist')]
    df = df.loc[:, ~df.columns.str.contains('edist')]

    route_map = ox.plot_route_folium(G2, route)
    HeatMap(df,
            radius=20,
            max_zoom=10,
            gradient={
                0.1: 'blue',
                0.2: 'lime',
                0.4: 'yellow',
                0.5: 'orange',
                0.7: 'red'
            }).add_to(route_map)

    filepath = 'route.html'
    route_map.save(filepath)


# value = 0
# selnode = 0
# print(points)
# pdist = 1000
# first = True
# for p in range(len(points)):
#     x = points[p].getX()
#     y = points[p].getY()
#     point = tuple((float(y), float(x)))
#     print(point)
#     selectedNode, dist = ox.get_nearest_node(Gnx, point, return_dist=True)
#     print(dist)
#     print(points[p].getValue())
#     if dist < 2000:
#         nodes['Pollution'][int(selectedNode)] = float(points[p].getValue())
#     if (float(dist) < float(pdist)) or first:
#         pdist = dist
#         selNode = selectedNode
#         value = points[p].getValue()
#     first = False
# print(selNode)
# nodes['Pollution'][int(selNode)] = float(value)
# for node in range(len(nodes)):
#     if nodes['Pollution'][node] > 0:
#         print(nodes['Pollution'][node])
tic = time.time()
nodes = set_values_to_nodes(points, nodes, Gnx)
toc = time.time()
print("Node processing lasted: " + str(toc-tic))
tic = time.time()
G = set_values_to_edges(points, edges, G)
toc = time.time()
print("Edge processing lasted: " + str(toc-tic))
Gnx = nx.relabel.convert_node_labels_to_integers(G)
ripnodes, edges = ox.graph_to_gdfs(Gnx, nodes=True, edges=True)
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
df.to_csv('points.csv')

#G2 = ox.graph_from_gdfs(set_values_to_nodes(points), edges)

# Terrassa coordinates
#destination_node = ox.get_nearest_node(G2,
#origin_node = ox.get_nearest_node(G2, (41.56538214481594, 2.010169694600521))
#                                       (41.56312304751537, 2.0302406158505706))
# Barcelona Coordinates
origin_node = ox.get_nearest_node(G2, (origin_yx))
destination_node = ox.get_nearest_node(G2, (destination_yx))
route = nx.shortest_path(G=G2,
                         source=origin_node,
                         target=destination_node,
                         weight='Pollution')
mapFolium(G2, route)
fig, ax = ox.plot_graph_route(
    G2,
    route,
    route_color="r",
    orig_dest_size=100,
    ax=None,
)