#%%
import osmnx as ox
import networkx as nx
import pandas as pd

G = ox.graph_from_place("Terrassa")
Gnx = nx.relabel.convert_node_labels_to_integers(G)
nodes, edges = ox.graph_to_gdfs(Gnx, nodes=True, edges=True)
#%%
d = pd.read_csv('points.csv')
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

d = {
    'node': node,
    'ndistance': ndistance
}
df = pd.DataFrame(d)

for ind in df.index:
    df['ndist'][ind].max