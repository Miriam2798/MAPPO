
#Vehicles:
#Pedestrian 0
#Bike 1
#Public transport 2
#E-Bike 3
#E-scooter 4
#E-motorbike 5
#E-car 6
#Hybrid Car 7
#Diesel/Gasoline car 8
#Mopeds and Motorcycle 9

#Route param: 
#R,G,W

def score(vehicle, route, score, km): #iterable i start
    """Score compute"""

    if(vehicle == 0 or vehicle == 1 or vehicle == 2):
        score += 60
    elif(vehicle == 3):
        score += 50
    elif(vehicle == 4):
        score += 40
    elif(vehicle == 5):
        score += 30
    elif(vehicle == 6):
        score += 20
    elif(vehicle == 7):
        score += 10
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
score0 = score(5,"W",50) 
print('score: ')
print(score0)

