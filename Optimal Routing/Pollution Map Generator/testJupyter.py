#%%
import osmnx as ox
import networkx as nx
import pandas as pd
import mappoAPI as api

G = ox.graph_from_place("vilafranca del penedes")
Gnx = nx.relabel.convert_node_labels_to_integers(G)
#%%
anodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
print(nodes)
nodes['Pollution'] = 0
#%%
# Gnx = nx.relabel.convert_node_labels_to_integers(G)
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
#%%
d = pd.read_csv('points_Vilafranca del Penedes.csv')
df = pd.DataFrame(d)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.sort_values(by=['node'])
#%%
pnode = 0
pedge = 0
ndist = 0
edist = 0
node = []
edge = []
ndistance = []
mean = []
edistance = []
first = True
for ind in df.index:
    if first:
        pnode = int(df['node'][ind])
        pndist = float(df['ndist'][ind])
        pedist = float(df['edist'][ind])
        first = False
    else:
        if pnode == df['node'][ind]:
            node.append(df['node'][ind])
            ndistance.append(df['ndist'][ind])
        else:
            pnode = df['node'][ind]
        if pedge == df['edge'][ind]:
            edge.append(df['edge'][ind])
            edistance.append(df['edist'][ind])
        else:
            pedge = df['edge'][ind]

d = {'node': node, 'ndistance': ndistance, 'pollution': node}
df = pd.DataFrame(d)
#%%
for ind in df.index:
    df['ndistance'][ind]
#%%
df = df.sort_values(by=['ndist'])
df = df.drop_duplicates(keep='first', subset='node')
nodes['Pollution'] = float(0)
for ind in df.index:
    nodes['Pollution'][int(df['node'][ind])] = 1 - df['value'][ind]
    print(nodes['Pollution'][int(df['node'][ind])])
print(nodes)
dfa = pd.DataFrame(nodes.drop(columns='geometry'))
#%%
df = df.sort_values(by=['edist'])
df = df.drop_duplicates(keep='first', subset='edge')
for ind in df.index:
    id = df['edge'][ind].split(",")
    id[0] = int(id[0].replace("(", ""))
    id[1] = int(id[1].replace(")", ""))
    G[id[0]][id[1]][0]['Pollution'] = 1 - df['value'][ind]
    print(id, ind, G[id[0]][id[1]][0]['Pollution'])
dfb = pd.DataFrame(G.drop(columns='geometry'))
#%%
import taxicab as tc

originx = 2.010169694600521
originy = 41.56538214481594
destinationx = 2.0302406158505706
destinationy = 41.56312304751537

origin_node = ox.get_nearest_node(G, tuple((originy, originx)))
destination_node = ox.get_nearest_node(G, tuple((destinationy, destinationx)))
#Finds the shortest path and stores it into the route object. Using length argument to find the shortest path in terms of length
route = nx.shortest_path(G=G,
                         source=origin_node,
                         target=destination_node,
                         weight='length')
routeTC = tc.distance.shortest_path(G, tuple((originy, originx)),
                                    tuple((destinationy, destinationx)))
print(route, routeTC)
#%%
d = pd.read_csv('points_Vilafranca del Penedes2.csv')
df = pd.DataFrame(d)
#nodes
df = df.sort_values(by=['ndist'])
df = df.drop_duplicates(keep='first', subset='node')
nodes['Pollution'] = float(0)
for ind in df.index:
    nodes['Pollution'][int(df['node'][ind])] = 1 - df['value'][ind]
a = pd.DataFrame(nodes)
df1 = pd.DataFrame(nodes.drop(columns='geometry'))

for source, target in G.edges():
    G[source][target][0]['Pollution'] = (nodes['Pollution'][source] +
                                         nodes['Pollution'][target]) / 2
#G.edges(data=True)
#%%
import pandas as pd

d = pd.read_csv('points_Vilafranca del Penedes.csv')
df = pd.DataFrame(d)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.loc[:, ~df.columns.str.contains('node')]
df = df.loc[:, ~df.columns.str.contains('edge')]
df = df.loc[:, ~df.columns.str.contains('ndist')]
df = df.loc[:, ~df.columns.str.contains('edist')]
#%%
import osmnx as ox
from mpl_toolkits.basemap import Basemap
import numpy as np

increment = 4
reso = 10
G = ox.graph_from_place("vilafranca del penedes")


#%%
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


#%%
import networkx as nx
from shapely.geometry import Point

G_projected = ox.project_graph(G)
nodes, edges = ox.graph_to_gdfs(G_projected, nodes=True, edges=True)
nodes['Pollution'] = float(0)
pollutionMatrix = np.loadtxt(open("map.csv", "rb"), delimiter=",")
rows = float(round(max(nodes['y']) - min(nodes['y']), increment))
cols = float(round(max(nodes['x']) - min(nodes['x']), increment))
incrow = round(rows / reso, increment)
incol = round(cols / reso, increment)
points = []
y = max(nodes['y'])
x = min(nodes['x'])

#%%
#Iteration that gives geolocation values to every value in the pollution matrix. Objects Point(y,x,value) are stored into a points list
for row in range(len(pollutionMatrix)):
    for col in range(len(pollutionMatrix[row])):
        if x > max(nodes['x']):
            x = round(min(nodes['x']), increment)
        points.append(
            Point(round(y, increment), round(x, increment),
                  pollutionMatrix[row][col]))
        x = x + incol
    if y <= round(min(nodes['y']), increment) and x >= round(
            max(nodes['x']), increment):
        break
    else:
        y = y - incrow

m = Basemap(projection='merc',
            llcrnrlat=min(nodes['y']),
            urcrnrlat=max(nodes['y']),
            llcrnrlon=min(nodes['x']),
            urcrnrlon=max(nodes['x']))
#%%
city = ox.geocode_to_gdf('vilafranca del penedes')
ax = ox.project_gdf(city).plot()
_ = ax.axis('off')
#%%
G_projected = ox.project_graph(G)
ox.plot_graph(G_projected)
ox.plot_graph(G)
#%%
import rasterio as rt
from rasterio.transform import Affine

x = np.linspace(min(nodes['x']), max(nodes['x']), 100)
y = np.linspace(min(nodes['y']), max(nodes['y']), 100)
X, Y = np.meshgrid(x, y)
Z = np.loadtxt(open("map.csv", "rb"), delimiter=",")
res = (x[-1] - x[0]) / 100
transform = Affine.translation(x[0] - res / 2, y[0] - res / 2) * Affine.scale(
    res, res)
new_dataset = rt.open(
    'new.tif',
    'w',
    driver='GTiff',
    height=Z.shape[0],
    width=Z.shape[1],
    count=1,
    dtype=Z.dtype,
    crs='+proj=latlong',
    transform=transform,
)
#%%
import osmnx as ox

G = ox.graph_from_place("vilafranca del penedes")
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
#%%
import pandas as pd
import ast
import math
import re

city = "vilafranca del penedes"

d = pd.read_csv('points_' + city + '.csv')
df = pd.DataFrame(d)
#nodes
df = df.sort_values(by=['ndist'])
df = df.drop_duplicates(keep='first', subset='node')
nodes['Pollution'] = float(0)
for ind in df.index:
    nodes['Pollution'][int(df['node'][ind])] = df['value'][ind]

dedges = pd.read_csv('edges_' + city + '.csv')
dfedges = pd.DataFrame(dedges)
dfedges = dfedges.loc[:, ~dfedges.columns.str.contains('^Unnamed')]
#%%
#tupledge = str(tuple((312213338, 386583494)))
#= 312213338
#= 386583494
dedges = pd.read_csv('edges_' + city + '.csv')
dfedges = pd.DataFrame(dedges)
dfedges = dfedges.loc[:, ~dfedges.columns.str.contains('^Unnamed')]
dfedges = dfedges.dropna(subset=['coords'])
for u, v, k in G.edges(keys=True):
    tupledge = str(tuple((u, v)))
    if tupledge in dfedges.values:
        coords = list(
            dfedges.loc[dfedges['edgeid'] == tupledge]['coords'].apply(
                ast.literal_eval))
        values = []
        if len(coords) != 0:
            for i in range(len(coords[0])):
                for index, row in df.iterrows():
                    if math.isclose(row['lat'],
                                    round(coords[0][i][1], 4),
                                    abs_tol=0.0004) and math.isclose(
                                        row['lon'],
                                        round(coords[0][i][0], 4),
                                        abs_tol=0.0004):
                        values.append(row['value'])
            G[u][v][k]['Pollution'] = sum(values) / len(values)
    else:
        G[u][v][k]['Pollution'] = (nodes['Pollution'][u] +
                                   nodes['Pollution'][v]) / 2
# %%
import networkx as nx

origin_yx = tuple((41.340791660130094, 1.6975732796178806))
destination_yx = tuple((41.352737557131356, 1.7011326493244299))
origin_node = ox.get_nearest_node(G, origin_yx)
destination_node = ox.get_nearest_node(G, destination_yx)
nx.shortest_path_length(G, origin_node, destination_node, weight='Pollution')
route = nx.shortest_path(G, origin_node, destination_node, weight='Pollution')
ox.plot.plot_graph_route(G, route)
#%%
import osmnx as ox
import networkx as nx
import pandas as pd
import ast
import math
import re
import time
import mappoAPI as api

G = ox.graph_from_place("vilafranca del penedes")
city = "vilafranca del penedes"
increment = 4
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
if os.path.exists('points_' + city + '.csv'):
    d = pd.read_csv('points_' + city + '.csv')
    df = pd.DataFrame(d)
    #nodes
    df = df.sort_values(by=['ndist'])
    df = df.drop_duplicates(keep='first', subset='node')
    nodes['Pollution'] = float(0)
    for ind in df.index:
        nodes['Pollution'][int(df['node'][ind])] = df['value'][ind]
    G = ox.graph_from_gdfs(nodes, edges)
    #edges
    dedges = pd.read_csv('edges_' + city + '.csv')
    dfedges = pd.DataFrame(dedges)
    dfedges = dfedges.loc[:, ~dfedges.columns.str.contains('^Unnamed')]
    dfedges = dfedges.dropna(subset=['coords'])
    for u, v, k in G.edges(keys=True):
        tupledge = str(tuple((u, v)))
        if tupledge in dfedges.values:
            coords = list(
                dfedges.loc[dfedges['edgeid'] == tupledge]['coords'].apply(
                    ast.literal_eval))
            values = []
            if len(coords) != 0:
                for i in range(len(coords[0])):
                    for index, row in df.iterrows():
                        if math.isclose(row['lat'],
                                        round(coords[0][i][1], increment),
                                        abs_tol=0.0004) and math.isclose(
                                            row['lon'],
                                            round(coords[0][i][0], increment),
                                            abs_tol=0.0004):
                            values.append(row['value'])
                G[u][v][k]['Pollution'] = (sum(values) / len(values))
        else:
            G[u][v][k]['Pollution'] = (nodes['Pollution'][u] +
                                       nodes['Pollution'][v]) / 2
    G2 = G
    toc = time.time()
#%%
ox.save_graphml(G2, "updated_graph.graphml")
G2 = ox.load_graphml("updated_graph.graphml",
                     node_dtypes={'Pollution': float},
                     edge_dtypes={'Pollution': float})
#%%
origin_yx = tuple((41.340791660130094, 1.6975732796178806))
destination_yx = tuple((41.352737557131356, 1.7011326493244299))
origin_node = ox.get_nearest_node(G2, origin_yx)
destination_node = ox.get_nearest_node(G2, destination_yx)
nx.shortest_path_length(G2, origin_node, destination_node, weight='Pollution')
# %%
route = nx.shortest_path(G2, origin_node, destination_node, weight='Pollution')
fastroute = nx.shortest_path(G=G2,
                             source=origin_node,
                             target=destination_node,
                             weight='length')
mixroute = nx.shortest_path(G=G2,
                            source=origin_node,
                            target=destination_node,
                            weight='mix')
filepath = 'routes.html'
rc = ['r', 'g', 'y']
ec = ox.plot.get_edge_colors_by_attr(G2, 'Pollution', cmap='autumn')
fig, ax = ox.plot_graph(G2, edge_color=ec, node_size=0, edge_linewidth=3)
fig, ax = ox.plot_graph_routes(G2, [fastroute, route, mixroute],
                               route_colors=rc,
                               route_linewidth=6)
filename = "lesspollutedroute.csv"
api.mapFolium(G2, route, fastroute, mixroute, filepath, origin_yx,
              destination_yx, city)
# %%
import osmnx as ox

G = ox.graph_from_place("vilafranca del penedes",network_type="bike")
ox.plot_graph(G)
