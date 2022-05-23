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
from shapely.geometry import shape
import math
import geopandas as gp
import ast
from networkx.classes.function import path_weight
import geopy.distance


class Point:
    """Defines a Point object with its own atributes Point(float y, float x, 
    float PollutionValue,int node, tuple edge)
    """

    # Constructor
    def __init__(self, y, x, value):
        self.y = y
        self.x = x
        self.value = value

    # ToString function
    def __repr__(self):
        return f"[{self.y}, {self.x}, {self.value}]"

    # Getters
    def getY(self):
        return self.y

    def getX(self):
        return self.x

    def getValue(self):
        return self.value

    def getNode(self):
        return self.node

    def getNdist(self):
        return self.ndist

    # Setters
    def setNode(self, node):
        self.node = node

    def setNdist(self, ndist):
        self.ndist = ndist


class Node:
    """Definition of a node object, with its attributes Node(float y, float x, int id, float value)
    """

    # Constructor
    def __init__(self, y, x, id, value):
        self.y = y
        self.x = x
        self.id = id
        self.value = value

    # ToString function. Returns the id, and y, x coordinates
    def __repr__(self):
        return f"{self.id}, {self.y}, {self.x},{self.value}"

    def getValue(self):
        return self.value


def removeWarnings():
    """Remove warnings
    """
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    warnings.simplefilter("ignore", UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def data_coord2view_coord(p, vlen, pmin, pmax):
    """Heatmap Plotting function

    Args:
        p (int): P parameter
        vlen (int): vlen parameter
        pmin (int): pmin parameter
        pmax (int): pmax parameter

    Returns:
        int: parameter
    """
    dp = pmax - pmin
    return (p - pmin) / dp * vlen


def nearest_neighbours(xs, ys, reso, n_neighbours):
    """Uses the nearest neighbor to smooth the cells and create a pollution simulation

    Args:
        xs (_type_): X parameter
        ys (_type_): Y parameter
        reso (_type_): Number of cells to generate (Resolution)
        n_neighbours (_type_): Nearest Neighbour factor

    Returns:
        Array of arrays: Array with pollution values
        Array: Array of border values
    """
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


def exportMap(im):
    """Export to csv file called map.csv

    Args:
        im (Array of arrays): Array with pollution values
    """
    np.savetxt("map.csv", im, delimiter=",")


def pollutionMapGenerator(generated_points, resolution, nn):
    """Generates an array of arrays with the given resolution and nearest neighbor factor. Uses the given generated points to do a gaussian mixture. 

    Args:
        generated_points (int): Number of desired points to be generaterd
        resolution (int): Number of cells to generate (Resolution)
        nn (int): Nearest Neighbour factor.
    """
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
    plt.show(block=False)
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
        "Please specify the number of points generated (100 to 10000) (default=1000): \n"
    )
    resolution = input(
        "Please specify the resolution (250 to 500) (default=100): \n")
    nn = input(
        "Please, specify the Nearest Neighbour parameter (default=16): \n")
    if generated_points == "":
        generated_points = 1000
    if resolution == "":
        resolution = 100
    if nn == "":
        nn = 16
    return generated_points, resolution, nn


def importFile(place, networktype):
    """If exists, imports the graph from the file. 
    The file is created after searching for a place once and for each one

    Args:
        place (string): Name of the city
        networktype (string): type of transport to use

    Returns:
        MultiDiGraph: Multi dimensional Graph containing nodes and edges
    """
    if os.path.exists(f'{place}_graph_'
                      f'{networktype}.txt'):
        return ox.load_graphml(f'{place}_graph_'
                               f'{networktype}.txt')
    print("\nFirst time running the script for " + place +
          ". Loading and Saving graph...\n")
    G = ox.graph_from_place(place, network_type=networktype, simplify=True)
    ox.save_graphml(G, f"{place}_graph_"
                    f"{networktype}.txt")
    return G


def export(G, routeTC, filename):
    """Exports the route into a route.csv file

    Args:
        G (MultiDiGraph): Multi dimensional Graph containing nodes and edges
        routeTC (List of int): list of nodes IDs
        filename (string): string with the name of the file

    Returns:
        List: returns a list with the coordinates, node ID and pollution values
    """
    nodelist = []
    # Iterate the nodes to extrat all the coordinates along with its ids
    for i in range(int(len(routeTC))):
        y = G.nodes[routeTC[i]]['y']
        x = G.nodes[routeTC[i]]['x']
        value = G.nodes[routeTC[i]]['Pollution']
        nodelist.append(Node(y, x, routeTC[i], value))
    # Create and write the node stored into nodelist to a route.csv file
    with open(filename, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in nodelist:
            writer.writerow([val])
    return nodelist


def dataMapping(origin_yx, destination_yx, city, reso, increment, networktype):
    """
    Pollution data projection depending on the given area. Coordinates are assignated to the 
    simulated pollution values.
    
    Parameters
    ----------
    origin_yx : tuple of two floats
            tuple of the latitude and longitude origin coordinates.
    destination_yx : tuple of two floats
            tuple of the latitude and longitude destination coordinates.
    city : string
            name of the place/city where we want to extract the graph.
    reso : int
            resolution of the pollution matrix. it has to be the same as the 
            value used for pollution matrix generation. a value of 100 means we
            have a 100x100 matrix in the pollution matrix, or 10000 cells.
    increment : int
            number of decimals to round the coordinates. example: with a value
            of 4, we are using 41.1234 instead of 41.123456789 
    networktype : string
            depending on our needs, we will extract a graph to commute by walk, car,
            bike, etc.
    Returns
    -------
    points : list of Points objects
            list of points with the given pollution values and their respective 
            coordinates
    """
    removeWarnings()
    G = importFile(city, networktype)
    gdf = ox.geocoder.geocode_to_gdf(city)
    ymax = float(gdf['bbox_north'])
    ymin = float(gdf['bbox_south'])
    xmax = float(gdf['bbox_east'])
    xmin = float(gdf['bbox_west'])
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    nodes['Pollution'] = float(0)
    pollutionMatrix = np.loadtxt(open("map.csv", "rb"), delimiter=",")
    rows = float(ymax - ymin)
    cols = float(xmax - xmin)
    incrow = rows / reso
    incol = cols / reso
    points = []
    y = float(round(ymax, increment))
    x = float(round(xmin, increment))
    # Iteration that gives geolocation values to every value in the pollution matrix.
    # Objects Point(y,x,value) are stored into a points list
    for row in range(len(pollutionMatrix)):
        for col in range(len(pollutionMatrix[row])):
            if x > round(xmax, increment):
                x = round(xmin, increment)
            points.append(
                Point(round(y, increment), round(x, increment),
                      pollutionMatrix[row][col]))
            x = x + incol
        if y < round(ymin, increment) and x > round(xmax, increment):
            break
        else:
            y = y - incrow
    values = []
    latitudes = []
    longitudes = []
    for p in range(len(points)):
        values.append(float(points[p].getValue()))
        latitudes.append(float(points[p].getY()))
        longitudes.append(float(points[p].getX()))
    data = {'value': values, 'Latitude': latitudes, 'Longitude': longitudes}
    df = pd.DataFrame(data)
    gdf = gp.GeoDataFrame(df,
                          geometry=gp.points_from_xy(df.Longitude,
                                                     df.Latitude))
    worldmap = gp.read_file(gp.datasets.get_path("naturalearth_lowres"))

    # Creating axes and plotting world map
    fig, ax = plt.subplots(figsize=(12, 6))
    worldmap.plot(color="lightgrey", ax=ax)

    # Plotting our Pollution data with a color map
    x = df['Longitude']
    y = df['Latitude']
    z = df['value']
    plt.scatter(x, y, s=20 * z, c=z, alpha=0.6, vmin=0, vmax=1, cmap='autumn')
    plt.colorbar(label='Pollution Values')

    # Creating axis limits and title
    plt.title("Pollution map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    return points


def set_values_to_edges(points, G, nodes, increment):
    """Gives Pollution values to edges. If an edge has geometry, it exports the coordinates of 
    the LineString that conforms the edge and calculates the average of pollution. For mix attribute, 
    it gives an average value between normalized length and pollution value

    Args:
        points (list of Points): list of points, with coordinates, node ID and pollution value
        G (MultiDiGraph): Multi dimensional Graph containing nodes and edges
        nodes (GeoDataFrame): GeoDataframe with nodes and their information
        increment (int): Number of decimals used for the coordinates (Latitude and Longitude number of decimals)

    Returns:
        MultiDiGraph: Multi dimensional Graph containing nodes and edges
        List: list with edge ID and its coordinates 
    """
    edges = []
    coords = []
    lengthmax = 0
    for u, v, k in G.edges(keys=True):
        if G[u][v][k]['length'] > lengthmax:
            lengthmax = G[u][v][k]['length']
        if not 'geometry' in G[u][v][k]:
            G[u][v][k]['Pollution'] = (
                (nodes['Pollution'][u] + nodes['Pollution'][v]) / 2)
            edges.append(tuple((u, v)))
            coords.append(None)
        else:
            a = list(shape(G[u][v][k]['geometry']).coords)
            edges.append(tuple((u, v)))
            coords.append(a)
            pollutionValues = []
            for p in range(len(points)):
                for i in range(len(a)):
                    if math.isclose(points[p].getY(),
                                    round(a[i][1], increment),
                                    abs_tol=0.0004) and math.isclose(
                                        points[p].getX(),
                                        round(a[i][0], increment),
                                        abs_tol=0.0004):
                        pollutionValues.append(float(points[p].getValue()))
                    else:
                        pollutionValues.append("")
            pollutionValues = [i for i in pollutionValues if i != ""]
            if len(pollutionValues) != 0:
                value = sum(pollutionValues) / len(pollutionValues)
            else:
                value = 1
            print("Edge " + str(G[u][v][k]) + "has a value of " + str(value) +
                  "\n")
            G[u][v][k]['Pollution'] = value
    data = {'edgeid': edges, 'coords': coords}
    for u, v, k in G.edges(keys=True):
        G[u][v][k]['length'] = G[u][v][k]['length'] / lengthmax
        mix = (G[u][v][k]['length'] * 0.5) + (G[u][v][k]['Pollution'] * 0.5)
        G[u][v][k]['mix'] = float(mix)
    return G, data


def set_values_to_nodes(points, nodes, G):
    """Checks what is the nearest point to a node using distances and assigns
    the pollution value to it.

    Args:
        points (list of Points): list of points, with coordinates, node ID and pollution value
        nodes (GeoDataFrame): GeoDataframe with nodes and their information
        G (MultiDiGraph): Multi dimensional Graph containing nodes and edges

    Returns:
        GeoDataFrame: Returns a geodataframe with nodes and their new attribute called Pollution
    """
    pdist = 0
    first = True
    nodes['Pollution'] = float(0)
    for p in range(len(points)):
        y = points[p].getY()
        x = points[p].getX()
        point = tuple((y, x))
        selectedNode, dist = ox.distance.nearest_nodes(G=G,
                                                       X=x,
                                                       Y=y,
                                                       return_dist=True)
        if first:
            pdist = dist
            first = False
        if dist <= pdist:
            print("Node " + str(selectedNode) + " has a value of " +
                  str(points[p].getValue()) + " and the nearest point is " +
                  str(point) + " at a distance of " + str(dist))
            nodes['Pollution'][selectedNode] = points[p].getValue()
        points[p].setNode(selectedNode)
        points[p].setNdist(dist)
    return nodes


def mapFolium(G2, route, fastroute, mixroute, filepath, originyx,
              destinationyx, city, networktype):
    """Export map as route.html using folium with 3 differents routes, and an overlay of a HeatMap
    representing the pollution in the area.

    Args:
        G2 (MultiDiGraph): Multi dimensional Graph containing nodes and edges
        route (int list): list of nodes IDs of the less polluted route
        fastroute (int list): list of nodes IDs of the shortest route
        mixroute (int list): list of nodes IDs of the combined route
        filepath (string): name and path of the file 
        originyx (float tuple): tuple with the origin coordinates
        destinationyx (float tuple): tuple with the destination coordinates
        city (string): name of the city
        networktype (string): type of transport to use
    """
    d = pd.read_csv('points_' + city + '_' + networktype + '.csv')
    df = pd.DataFrame(d)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.loc[:, ~df.columns.str.contains('node')]
    df = df.loc[:, ~df.columns.str.contains('ndist')]

    route_map = ox.plot_route_folium(G2, route, route_color='#32CD32')
    route_map = ox.plot_route_folium(G2,
                                     fastroute,
                                     route_map=route_map,
                                     route_color='#ff0000')
    route_map = ox.plot_route_folium(G2,
                                     mixroute,
                                     route_color='#ffff00',
                                     route_map=route_map)
    HeatMap(data=df, radius=15, max_zoom=13).add_to(route_map)
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
    city = input(
        "Please, insert the place name: (DEFAULT: Vilafranca del Penedes) \n")
    origin_yx = input(
        "Please, insert the origin coordinates: (DEFAULT: 41.59047, 2.45235) \n"
    )
    destination_yx = input(
        "Please, insert the destination coordinates: (DEFAULT: 41.59047, 2.45235) \n"
    )
    if origin_yx != "" and destination_yx != "":
        originy, originx = origin_yx.split(", ")
        destinationy, destinationx = destination_yx.split(", ")
        origin_yx = tuple((originy, originx))
        destination_yx = tuple((destinationy, destinationx))
    update = input("Do you want to update pollution values? (DEFAULT: no)\n")
    networktype = input(
        "\nChoose your vehicle: (DEFAULT: drive)\n\nallprivate (ap)\nall (a)\nbike (b)\ndrive (d)\ndriveservice (ds)\nwalk(w)\n\n"
    )
    if city == "":
        city = "Vilafranca del Penedes"
    if origin_yx == "":
        origin_yx = tuple((41.34966753431925, 1.6959634103837375))
    if destination_yx == "":
        destination_yx = tuple((41.33632345834082, 1.6946716922151437))
    if networktype == "ap":
        networktype = "all_private"
    elif networktype == "a":
        networktype = "all"
    elif networktype == "b":
        networktype = "bike"
    elif networktype == "d":
        networktype = "drive"
    elif networktype == "ds":
        networktype = "drive_service"
    elif networktype == "w":
        networktype = "walk"
    elif networktype == "":
        networktype = "drive"
    else:
        print(
            "\nPlease, enter a valid option or read the documentation for more information.\n"
        )
    if update == "yes":
        update = True
    else:
        update = False
    G = importFile(city, networktype)
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    return city, origin_yx, destination_yx, nodes, edges, G, update, networktype


def importFileFromPoint(originy, originx, destinationy, destinationx,
                        networktype):
    """Imports a partitioned graph given the coordinates of the origin and destination
    points of the route. Currently not used, but created for further implementation

    Args:
        originy (float): latitude of the origin point
        originx (float): longitude of origin point
        destinationy (float): latitude of the destination point
        destinationx (float): longitude of the destination point
        networktype (string): type of transport to use
    """
    center_pointx = (originx + destinationx) / 2
    center_pointy = (originy + destinationy) / 2
    center_point = tuple((center_pointy, center_pointx))
    distance = geopy.distance.geodesic(center_point, tuple(
        (originy, originx))).m + 1000
    G5 = ox.graph_from_point(center_point,
                             distance,
                             dist_type="bbox",
                             network_type=networktype)
    fig, ax = ox.plot_graph(G5, node_size=0, edge_linewidth=3)


def updateValues(originx, originy, destinationx, destinationy, city, reso,
                 increment, G, networktype):
    """In order to save computation time, this functions checks if points.csv and edges.csv 
    exist and updates them. If not, it creates them.

    Args:
        originx (float): longitude of origin point
        originy (float): latitude of origin point
        destinationx (float): longitude of destination point
        destinationy (float): latitude of destination point
        city (string): name of the city
        reso (int): resolution of the pollution matrix
        increment (int): Number of decimals used for the coordinates (Latitude and Longitude number of decimals)
        G (MultiDiGraph): Multi dimensional Graph containing nodes and edges
        networktype (string): type of transport to use
    """
    removeWarnings()
    tic = time.time()
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    if os.path.exists('points_' + city + '_' + networktype +
                      '.csv') and os.path.exists('edges_' + city + '_' +
                                                 networktype + '.csv'):
        d = pd.read_csv('points_' + city + '_' + networktype + '.csv')
        df = pd.DataFrame(d)
        # nodes
        df = df.sort_values(by=['ndist'])
        df = df.drop_duplicates(keep='first', subset='node')
        nodes['Pollution'] = float(0)
        for ind in df.index:
            nodes['Pollution'][int(df['node'][ind])] = df['value'][ind]
        G = ox.graph_from_gdfs(nodes, edges)
        # edges
        dedges = pd.read_csv('edges_' + city + '_' + networktype + '.csv')
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
                                                round(coords[0][i][0],
                                                      increment),
                                                abs_tol=0.0004):
                                values.append(row['value'])
                    if len(values) != 0:
                        G[u][v][k]['Pollution'] = (sum(values) / len(values))
                    else:
                        G[u][v][k]['Pollution'] = 0.5
            else:
                G[u][v][k]['Pollution'] = (nodes['Pollution'][u] +
                                           nodes['Pollution'][v]) / 2
            G[u][v][k]['mix'] = (
                G[u][v][k]['Pollution'] +
                (G[u][v][k]['length'] / edges['length'].max())) / 2
        G2 = G
        toc = time.time()
        print("Node & Edge importing lasted: " + str(toc - tic))
    else:
        origin_yx = tuple((originy, originx))
        destination_yx = tuple((destinationy, destinationx))
        print("\nFirst time running the script. Mapping the data...\n")
        points = dataMapping(origin_yx, destination_yx, city, reso, increment,
                             networktype)
        tic = time.time()
        nodes = set_values_to_nodes(points, nodes, G)
        toc = time.time()
        print("Node processing lasted: " + str(toc - tic))
        tic = time.time()
        G, data = set_values_to_edges(points, G, nodes, increment)
        toc = time.time()
        print("Edge processing lasted: " + str(toc - tic))
        ripnodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
        G2 = ox.graph_from_gdfs(nodes, edges)
        dfedges = pd.DataFrame(data)
        dfedges.to_csv('edges_' + city + '_' + networktype + '.csv')
        lat = []
        lon = []
        val = []
        node = []
        ndist = []
        for p in range(len(points)):
            lat.append(points[p].getY())
            lon.append(points[p].getX())
            val.append(points[p].getValue())
            node.append(points[p].getNode())
            ndist.append(points[p].getNdist())
        d = {
            'lat': lat,
            'lon': lon,
            'value': val,
            'node': node,
            'ndist': ndist,
        }
        df = pd.DataFrame(d)
        df.to_csv('points_' + city + '_' + networktype + '.csv')
    ox.save_graphml(G2,
                    "updated_graph_" + city + "_" + networktype + ".graphml")


def routesComputing(originy, originx, destinationy, destinationx, city,
                    networktype):
    """
    Calculates 3 different routes:
    
    -Shortest route: Computes the shortest path from point A to B using Dijkstra's
    algorithm, given the edge length as weight.
    
    -Combined route: Computes the shortest path from point A to B using Dijkstra's
    algorithm, given a combination of 2 edge parameters (the edge length and pollution amount).
    
    -Less pollution exposed route: Computes the shortest path from point A to B using Dijkstra's
    algorithm, given the edge pollution amount as weight.
    
    Parameters
    ----------
    originy : float
            latitude coordinate of origin point
    originx : float
            longitude coordinate of origin point
    destinationy : float
            latitude coordinate of the destination point
    destinationx : float
            longitude coordinate of the destination point
    city : string
            name of the place/city where we want to extract the graph
    
    Returns
    -------
    export(G2, route, "lesspollutedroute.csv") : function call for the exportation of a nodelist
            given the less polluted route. Creation of a csv file called lesspollutedroute.csv
    export(G2, route, "fastestroute.csv") : function call for the exportation of a nodelist
            given the shortest route. Creation of a csv file called fastestroute.csv
    export(G2, route, "mixroute.csv") : function call for the exportation of a nodelist
            given the combined route. Creation of a csv file called mixroute.csv
    """
    removeWarnings()
    G2 = ox.load_graphml("updated_graph_"
                         f"{city}_"
                         f"{networktype}.graphml",
                         node_dtypes={
                             'Pollution': float,
                             'mix': float
                         },
                         edge_dtypes={
                             'Pollution': float,
                             'mix': float
                         })
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
    mixroute = nx.shortest_path(G=G2,
                                source=origin_node,
                                target=destination_node,
                                weight='mix')
    lesspollutedpathweight = path_weight(G2, route, weight="Pollution")
    fastroutepathweight = path_weight(G2, fastroute, weight="Pollution")
    mixedpathweight = path_weight(G2, mixroute, weight="Pollution")
    filepath = 'route.html'
    rc = ['r', 'g', 'y']
    ec = ox.plot.get_edge_colors_by_attr(G2, 'Pollution', cmap='autumn')
    fig, ax = ox.plot_graph(G2, edge_color=ec, node_size=0, edge_linewidth=3)
    fig, ax = ox.plot_graph_routes(G2, [fastroute, route, mixroute],
                                   route_colors=rc,
                                   route_linewidth=6)
    mapFolium(G2, route, fastroute, mixroute, filepath, origin_yx,
              destination_yx, city, networktype)
    return export(G2, route, "lesspollutedroute.csv"), export(
        G2, fastroute, "fastestroute.csv"), export(
            G2, mixroute, "mixroute.csv"
        ), lesspollutedpathweight, fastroutepathweight, mixedpathweight
