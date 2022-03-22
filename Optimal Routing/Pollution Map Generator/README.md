# MAPPO

## Optimal Routing
### Installation
In order to use the pollution maps generator you must import the modules required using the following command line:
```
pip install -r requirements.txt
```
### Usage
Just run it with Python 3.10.0 using the following command line:
```
python3 menu.py
```
A menu will show up, with different options to select

### Hint
All the different points are explained below, along the individual installation and usage procedures

## 1. Pollution Maps Generator
### Installation
In order to use the pollution maps generator you must import the modules required using the following command line:
```
pip install -r requirements.txt
```
### Usage 
Just run it with Python 3.10.0 using the following command line:
```
python3 pollutionmap.py
```
Default parameters are suggested before inputing the desired number.

## Default parameters
The script will ask for some input parameters. First input are the desired generated points. As it is stated, default option is 1000 points, so it generates 1000x1000 points, whose will be positions in the matrix. Secondly, it is the resolution, which determines the pixel dimension in the output image. It is 500 by default. Last but not least, it is the Nearest Neighbour factor, which is a parameter that will change the way how the points are joint into cluster values. By default it is recommended to use 16. 

Default values are advised as the final output may vary depending on the parameters. In order to adjust as much as possible to reality, we will use the ones mentioned before, as in our opinion, are more realistic than other tested parameters. Also, increasing generated points will not affect the output of the simulation but the computational speed. You may try higher numbers, but the output time will be higher.

### Output
After running the script, a file called `map.csv` should appear. This file has all the values stored in a two dimensional matrix.
The script will also show the heatmap of the actual generated pollution map.

## 2. Transform map
### Installation
In order to use the pollution maps generator you must import the modules required using the following command line:
```
pip install -r requirements.txt
```
### Usage 
Just run it with Python 3.10.0 using the following command line:
```
python3 transformap.py
```
Default parameters are suggested before inputing the desired number.

### Output
After running the script, a file called `bordermap.csv` should appear. This file has all the values stored in a two dimensional matrix.
The script will show a binary map of the values that are taken into account depending on the threshold value

## 3. Fastest Path
### Installation
In order to use the pollution maps generator you must import the modules required using the following command line:
```
pip install -r requirements.txt
```
### Usage 
Just run it with Python 3.10.0 using the following command line:
```
python3 fastestpath.py
```
Default parameters are suggested before inputing the desired number.

### Output
After running the script, it will show a the actual graph map with all the nodes and the optimal fastest route depending on lenght.