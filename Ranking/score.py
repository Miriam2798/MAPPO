
#Vehicles:
#Pedestrian 0
#Bike 1
#Public transport 2
#E-Bike 3
#E-scooter 4
#E-motorbike 5
#E-car 6
#Hybrid Car 7
#Diesel car 8
#Gasoline car 9
#Mopeds and Motorcycle 9

#Route param: 
#R,G,W

def score(vehicle, route, score, km): #iterable i start
    """Score compute"""
    if(vehicle == 0 or vehicle == 1 or vehicle == 2 or vehicle == 3 or vehicle == 4 or vehicle == 5 or vehicle == 6):
        score += 40
    elif(vehicle == 7):
        score += 37
    elif(vehicle == 8):
        score += 32
    elif(vehicle == 9):
        score += 21
    else:
        score += 0

    if(route == "W"):
        score += 10
    elif(route == "G"):
        score += 5
    else: 
        score += 0
    score = score*km

    return score
#pruebas
score0 = score(5,"W",50,1) 
print('score: ')
print(score0)
