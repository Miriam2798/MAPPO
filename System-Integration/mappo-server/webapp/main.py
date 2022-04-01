from importlib.resources import path
import requests
from app import app



r= requests.get('https://mappo-server.herokuapp.com/')

with open(r'/usr/src/app/mappo-server/webapp/prova.txt', 'wb') as f:
    f.write(r.content)
place = "Barcelona"

app.run()


