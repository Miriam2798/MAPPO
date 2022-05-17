import os
import requests

#os.environ['NO_PROXY'] = '127.0.0.1'
option_routing = "fastest"
place = "vilafranca%20del%20penedes"
originx = "41.33821155058145"
originy = "1.6916121031650182"
destinationx = "41.35009126561326"
destinationy = "1.7012326058103266"
option_db = "rankthismonthuser"
ruta = requests.get('http://mappo-server.herokuapp.com/routing?option='+ option_routing +'&place='+ place +'&originx=' + originx 
                    +'&originy=' + originy + '&destinationx=' + destinationx + '&destinationy=' + destinationy)

ruta = str(ruta.content)

#print(ruta)

contno2 = requests.get('http://mappo-server.herokuapp.com/database?option=contno2')

rankthismonthuser= requests.get('http://mappo-server.herokuapp.com/database?option=rankthismonthuser')
rankthismonthuser = str(rankthismonthuser.content)

print(rankthismonthuser)

contno2 = str(contno2.content)


primer_caracter = False
last_char = ''

with open('mi_fichero.csv', 'w') as f:

    for c in contno2:
        if c == '(':
            primer_caracter = True
            
        if (not (c < '0' or c > '9' and c < 'A' or c > 'Z' and c < 'a' or c > 'z') or c == '.' or c == '-') and primer_caracter==True:
            last_char = c
            f.write(c)

        if c == ')':
            f.write(',\n')
            last_char = c

        if c == ',' and last_char != ')' and primer_caracter == True:
            f.write(c)

            


