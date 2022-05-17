import fastestpath as fp
import pollutionmap as pm
import transformap as tm
import time
import os
import mappoAPI as api
import networkx as nx
import osmnx as ox
from termcolor import colored


def filesCheck(city, networktype):
    if os.path.exists('points_' + city + '_' + networktype +
                      '.csv') and os.path.exists('edges_' + city + '_' +
                                                 networktype + '.csv'):
        pointsEdges = True
    else:
        pointsEdges = False
    if os.path.exists('map.csv'):
        pollutionMap = True
    else:
        pollutionMap = False
    if os.path.exists("updated_graph_"
                      f"{city}_"
                      f"{networktype}.graphml"):
        updated = True
    else:
        updated = False
    return pollutionMap, pointsEdges, updated


def menu():
    print(r"""   _____      _____ ____________________________   
  /     \    /  _  \\______   \______   \_____  \  
 /  \ /  \  /  /_\  \|     ___/|     ___//   |   \ 
/    Y    \/    |    \    |    |    |   /    |    \
\____|__  /\____|__  /____|    |____|   \_______  /
        \/         \/                           \/ v0.3 """)
    time.sleep(1)
    print("1. Simulate Pollution Map \n")
    print("2. Calculate the 3 routes \n")
    print("3. Exit \n")


loop = True
while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    menu()
    choice = input("Enter your choice [1-3]: \n")
    os.system('cls' if os.name == 'nt' else 'clear')
    if choice == "1":
        generated_points, reso, nn = api.mainPollutionMapGenerator()
        api.pollutionMapGenerator(generated_points, reso, nn)
    elif choice == '2':
        city, origin_yx, destination_yx, nodes, edges, G, update, networktype = api.mainLessPollutedRoute(
        )
        pollutionMap, pointsEdges, updated = filesCheck(city, networktype)
        os.system('cls' if os.name == 'nt' else 'clear')
        if pollutionMap:
            print(colored('\n (OK) Pollution Map file (map.csv)', 'green'))
        else:
            print(
                colored('\n (FAIL) Pollution Map file (map.csv is missing)',
                        'red'))
            print(
                "\n Please, simulate the map using the first option in the menu. "
            )
        if pointsEdges:
            print(
                colored(
                    '\n (OK) Nodes and Edges files (points_city_networktype.csv & edges_city_networktype.csv)',
                    'green'))
        else:
            print(
                colored(
                    '\n (FAIL) Nodes and Edges files (points_city_networktype.csv & edges_city_networktype.csv) are missing',
                    'red'))
        if updated:
            print(
                colored(
                    '\n (OK) Edge update file (updated_graph_city_networktype.csv)',
                    'green'))
        else:
            print(
                colored(
                    '\n (FAIL) Edge update file (updated_graph_city_networktype.csv) is missing',
                    'red'))
            print(
                "\n You said no to updating pollution values but it appears you don't have a updated file yet. We will update it for you...\n"
            )
        if pollutionMap and updated and pointsEdges:
            print(colored('\n ALL REQUIRED FILES FOUND. PROCEEDING...'))
        exitwait = input("\n PRESS ANY KEY TO PROCEED\n ")
        os.system('cls' if os.name == 'nt' else 'clear')
        if update or not updated and pollutionMap:
            api.updateValues(origin_yx[0], origin_yx[1], destination_yx[0],
                             destination_yx[1], city, 100, 4, G, networktype)
            print("Pollution values updated. \n")
        elif pollutionMap:
            nodelistpolluted, nodelistfast, nodelistmix = api.routesComputing(
                origin_yx[0], origin_yx[1], destination_yx[0],
                destination_yx[1], city, networktype)
            lesspollutedsum = 0
            fastsum = 0
            mixsum = 0
            for i in range(len(nodelistpolluted)):
                lesspollutedsum += nodelistpolluted[i].getValue()
            for i in range(len(nodelistfast)):
                fastsum += nodelistfast[i].getValue()
            for i in range(len(nodelistmix)):
                mixsum += nodelistmix[i].getValue()
            print("\n(GREEN ROUTE) Less Pollute Route Pollution: " +
                  str(lesspollutedsum) + "\n")
            print("\n(YELLOW ROUTE) Combined Pollution/distance Pollution: " +
                  str(mixsum) + "\n")
            print("\n(RED ROUTE) Shortest Route Pollution: " + str(fastsum) +
                  "\n")
            print("\nExposure to Pollution reduction (GREEN ROUTE): " +
                  str(round(100 -
                            ((lesspollutedsum / fastsum) * 100), 2)) + "%\n")
            print("Exposure to Pollution reduction (YELLOW ROUTE):" +
                  str(round(100 - ((mixsum / fastsum) * 100), 2)) + "%\n")
        exitwait = input("Press any key...")
    elif choice == "3":
        loop = False
        os.system('cls' if os.name == 'nt' else 'clear')
        break
    else:
        input("Wrong Option. Press any key to try again")