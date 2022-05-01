# MAPPO

![MAPPO](https://github.com/annapuig/MAPPO/blob/main/Pictures/mappo.jpg)

## System Integration
### Usage of the server
In order to use the functions that runs in the server (in python), you must follow this guide.

## 1. Generate a python script that can be named test.py

### Routing
From this script, we will obtain the outputs of the functions that run in the server. An example can be

```
import requests

option_routing = "fastest"
place = "vilafranca%20del%20penedes"
originx = "41.33821155058145"
originy = "1.6916121031650182"
destinationx = "41.35009126561326"
destinationy = "1.7012326058103266"

route = requests.get('http://mappo-server.herokuapp.com/routing?option='+ option_routing +'&place='+ place +'&originx=' + originx 
                    +'&originy=' + originy + '&destinationx=' + destinationx + '&destinationy=' + destinationy)
                    
```
This request will make the server to run a script which returns a route (the fastest in this case) between point A defined by originx and 
originy to point B which is defined by destinationx and destinationy.

For the routing index, there are two options which are:
  - fastest
  - lesspolluted


Each of these, returns a different route. The first returns the fastest and the second the less polluted one. We can access to one or another 
by changing the value of option_routing in the script above.



## Database

Also, we can acces to some functions that returns some values from the database, an example is

```
import requests

option_db = "userdata"
userdata= requests.get('http://mappo-server.herokuapp.com/database?option=' + option_db)
userdata = str(userdata.content)


```

we can custom the return by changing the value of option_db by:
  - rankthismonthuser
  - userdata
  - rank
  - contno2


Rankthismonthuser is a string which has a ranking with every user sorted from largest to smallest punctuations of this month. 
Userdata, returns all the user registered into our database. Rank, returns the global ranking also sorted from largest to smallest. 
Contno2, returns the coordinates and the station that collects the NO2 particles.


### Output

For routing

```
b'[446285149, 41.3381889, 1.6916727,0.860052710319143, 446285150, 41.3383107, 1.6917474,0.8576286224352969, 
1650980469, 41.338629, 1.6922612,0.7483626933131706, 446285155, 41.3389308, 1.6927853,0.0, 446285156, 41.3389072, 1.6928587,0.7249362006531911, 
446285152, 41.3390074, 1.6929092,0.7034209674118209, 446285163, 41.3400776, 1.6947437,0.7221006145282836, 395310270, 41.3407369, 1.6943844,0.8029070354073253, 
395310268, 41.3408805, 1.6943214,0.0, 4557494476, 41.3411492, 1.6941807,0.9069667451066141, 489649164, 41.341216, 1.694141,0.0, 
933328733, 41.3414237, 1.6944493,0.9148676585404457, 1115784676, 41.3419608, 1.695192,0.8054158276244311, 395310285, 41.3424845, 1.6959161,0.6726681950305591, 
1650980522, 41.3434649, 1.6972719,0.4599062966133488, 1653464744, 41.3440495, 1.6980803,0.5908455040557781, 1650980526, 41.3443133, 1.6984451,0.703666620870106, 
361495245, 41.3444274, 1.6986028,0.7099585168355921, 795816212, 41.3447963, 1.6993281,0.9582039413199064, 361495241, 41.3457482, 1.7012787,0.6055605319924998, 
1653464951, 41.3459413, 1.7010585,0.648731337447196, 1653464976, 41.3461164, 1.7012239,0.35818609088229725, 1653465050, 41.3466572, 1.7009447,0.30264435058040007, 
1653465078, 41.3468025, 1.7007016,0.2516658016579184, 1653465328, 41.3481554, 1.701027,0.0, 1653465362, 41.3482437, 1.7010975,0.7267483704104024, 
489649237, 41.3484103, 1.7012304,0.7405301379457994, 1653465461, 41.3489159, 1.7000174,0.9097120266538411, 
1653465555, 41.3493998, 1.7002951,0.7679288447929002, 1653465607, 41.3497497, 1.7004736,0.7129395667921727, 395310364, 41.3503564, 1.7006777,0.5558826001976129]'
```
And for the database
```
b"[(1, 1, 'Marc Ensesa Planellas', 'GreenMarc', 'marc@gmail.com', 22, 'Gracia', '1234'), (2, 1, 'Pau Villaverde', 'pau', 'pauvima15@gmail.com', 21, None, '12345'), 
(3, 2, 'Guillem', 'Guille', 'guille@gmail.com', None, None, '$2a$08$fnKY5BYm/zbp9t.hO9AARONcE.JbOx67clAb/ABlOvQ2or66v7nWC'), 
(6, None, 'aaaa', 'aaaa', 'aaaa@s.es', None, None, '12345'), (28, None, 'sergi', 'sroura', 'sergi', 0, None, '12345'), 
(4, 2, 'Miriam', 'M', 'miriam@gmail.com', 23, 'Clot', '$2a$08$FnvlSpIbpPNqrka3sN1nrO/iXM6J2SJwZKtrZt3M8ENdoOsi21Dxy'), 
(30, None, 'anna', 'anna', 'anna@gmail.com', 0, None, '1234'), 
(5, 2, 'Pep', 'Pep', 'pepito@gmail.co,', None, None, '$2a$08$NFK6Xhqj1gyOfvz58NQHM.cMeNuf/QnaiBO37Tfrb4/hRiN21copa'), 
(31, None, 'mmm', 'mmm', 'mmm', 0, None, 'mmm'), (13, None, 'ss', 'ss', 'ss', 0, None, 'ss'), (14, None, 'ss', 'jjjj', '', 0, None, 'ss'), 
(15, None, 'RegisterTest', 'registerTest', 'registerText@gmail.com', 0, None, '12345'), (16, None, 'dd', 'dd', 'dd', 0, None, 'dd'), 
(17, None, 'aad', 'aad', 'asd', 0, None, 'aad'), (22, None, 'Anna', 'apuig', 'apuig@gmail.com', 0, None, '1234'), 
(23, None, 'juanito', 'juanito', 'juan', 0, None, '12345'), (25, None, 'Mamemi', 'Ma', 'Maa@gmail.com', 24, 'Gracia', '1234'), (26, 0, 'Testsss', 'T', 'T@', 0, 'BCN', '1234')]"

```
