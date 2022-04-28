import fastestpath as fp
import pollutionmap as pm
import transformap as tm
import time
import os
import mappoAPI as api
import networkx as nx
import osmnx as ox


def menu():
    print(r"""   _____      _____ ____________________________   
  /     \    /  _  \\______   \______   \_____  \  
 /  \ /  \  /  /_\  \|     ___/|     ___//   |   \ 
/    Y    \/    |    \    |    |    |   /    |    \
\____|__  /\____|__  /____|    |____|   \_______  /
        \/         \/                           \/ v0.3 """)
    time.sleep(1)
    print("1. Simulate Pollution Map \n")
    print("2. Transform Pollution Map (DEPRECATED) \n")
    print("3. Calculate Fastest Route \n")
    print("4. Calculate Less Polluted Route \n")
    print("5. Exit \n")


loop = True
while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    menu()
    choice = input("Enter your choice [1-5]: \n")
    os.system('cls' if os.name == 'nt' else 'clear')
    if choice == "1":
        pm.main()
    elif choice == "2":
        tm.main()
    elif choice == "3":
        fp.main()
    elif choice == '4':
        city, origin_yx, destination_yx = api.mainLessPollutedRoute()
        G = api.importFile(city)
        Gnx = nx.relabel.convert_node_labels_to_integers(G)
        nodes, edges = ox.graph_to_gdfs(Gnx, nodes=True, edges=True)
        api.LessPollutedRoute(origin_yx[1], origin_yx[0], destination_yx[1],
                              destination_yx[0], city, 10, 4, nodes, edges, G)
    elif choice == "5":
        loop = False
        os.system('cls' if os.name == 'nt' else 'clear')
        break
    else:
        input("Wrong Option. Press any key to try again")