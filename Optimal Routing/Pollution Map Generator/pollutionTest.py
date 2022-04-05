# TEST SCRIPT, DO NOT TRY TO USE LOL
import warnings
from pandas.core.common import SettingWithCopyWarning

import osmnx as ox
import networkx as nx
import numpy as np
import math
import fastestpath as fp
import warnings
import multiprocessing

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
        return f"[{self.y}, {self.x}, {self.value}]"

    #Getters
    def getY(self):
        return self.y

    def getX(self):
        return self.x

    def getValue(self):
        return self.value


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
edges['Pollution'] = float(0)

for edge in range(len(edges)):
    edges['key']=edge

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
def set_values_to_edges(points):
    edist = 0
    first = True
    for p in range(len(points)):
        y = points[p].getY()
        x = points[p].getX()
        point = tuple((y, x))
        #selectedEdge, edist = ox.nearest_edges(G, x, y, return_dist=True)
        u, v, a, edist = ox.get_nearest_edge(G, point, return_dist=True)
        if first:
            epdist = edist
            first = False
        if edist <= epdist:
            print("Edge " + str(u) + " " + str(v) + " " + str(a) +
                  " has a value of " + str(points[p].getValue()) +
                  " and the nearest point is " + str(point) +
                  " at a distance of " + str(edist) + "\n")
            edges['Pollution'][a] = 1 - points[p].getValue()
    return edges


def set_values_to_nodes(points):
    pdist = 0
    first = True
    for p in range(len(points)):
        y = points[p].getY()
        x = points[p].getX()
        point = tuple((y, x))
        selectedNode, dist = ox.get_nearest_node(Gnx, point, return_dist=True)
        if first:
            pdist = dist
            first = False
        if dist <= pdist:
            print("Node " + str(selectedNode) + " has a value of " +
                  str(points[p].getValue()) + " and the nearest point is " +
                  str(point) + " at a distance of " + str(dist))
            nodes['Pollution'][selectedNode] = 1 - points[p].getValue()
    return nodes


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
edges=set_values_to_edges(points)
print(edges)
G2 = ox.graph_from_gdfs(set_values_to_nodes(points),
                        edges)
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
route_map=ox.plot_route_folium(G, route)
filepath = 'route.html'
route_map.save(filepath)
fig, ax = ox.plot_graph_route(
    G2,
    route,
    route_color="r",
    orig_dest_size=100,
    ax=None,
)