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
        #pm.main()
        generated_points, reso, nn = api.mainPollutionMapGenerator()
        api.pollutionMapGenerator(generated_points, reso, nn)
    elif choice == "2":
        tm.main()
    elif choice == "3":
        #fp.main()
        city, origin_yx, destination_yx = api.mainFastestRoute()
        api.fastest_route(origin_yx[0], origin_yx[1], destination_yx[0],
                          destination_yx[1], city)
    elif choice == '4':
        city, origin_yx, destination_yx, nodes, edges, G = api.mainLessPollutedRoute(
        )
        os.system('cls' if os.name == 'nt' else 'clear')
        nodelistpolluted, nodelistfast = api.LessPollutedRoute(
            origin_yx[1], origin_yx[0], destination_yx[1], destination_yx[0],
            city, 200, 5, nodes, edges, G)
        lesspollutedsum = 0
        fastsum = 0
        for i in range(len(nodelistpolluted)):
            lesspollutedsum += nodelistpolluted[i].getValue()
        for i in range(len(nodelistfast)):
            fastsum += nodelistfast[i].getValue()
        print("\nLess Pollute Route Pollution: " + str(lesspollutedsum) + "\n")
        print("Shortest Route Pollution: " + str(fastsum) + "\n")
        print("Exposure to Pollution reduction: " +
              str(round(100 - ((fastsum / lesspollutedsum) * 100), 2)) + "%\n")
        exitwait = input("")
    elif choice == "5":
        loop = False
        os.system('cls' if os.name == 'nt' else 'clear')
        break
    else:
        input("Wrong Option. Press any key to try again")