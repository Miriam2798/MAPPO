#%%
import osmnx as ox
import networkx as nx
import pandas as pd
import mappoAPI as api

G = ox.graph_from_place("Terrassa")
#%%
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
#print(nodes)
dfa = pd.DataFrame(nodes.drop(columns='geometry'))
#%%
df = df.sort_values(by=['edist'])
df = df.drop_duplicates(keep='first', subset='edge')
for ind in df.index:
    id = df['edge'][ind].split(",")
    id[0] = int(id[0].replace("(", ""))
    id[1] = int(id[1].replace(")", ""))
    G[id[0]][id[1]][0]['Pollution'] = 1 - df['value'][ind]
    print(id,ind,G[id[0]][id[1]][0]['Pollution'])
dfb = pd.DataFrame(G.drop(columns='geometry'))
#%%
print(G[1126668145][282873806])