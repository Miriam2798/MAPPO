import numpy
import pandas
import psycopg2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
#No se si es necesario:
from sklearn.preprocessing import MinMaxScaler
from data_forecast import data




conn = psycopg2.connect(user="feflopfeklpznc",
                                    password="5de7e5b5fc9f83e323359f1c4ba05394ed23356dd8ba561aa45b88b54d11c026",
                                    host="ec2-54-73-167-224.eu-west-1.compute.amazonaws.com",
                                    port="5432",
                                    database="dfi7i5f0k4mkd2")
numpy.random.seed(7)
#dataset = pandas.read_csv('nom', usecols=[1]?, engine='python')

dataset = pandas.read_sql_query("""
                        SELECT nom_estacio, "data", contaminant, unitats, latitud, longitud, no2
                        FROM definitiu.no2_csv

                        """, conn)
df= pandas.DataFrame(dataset, columns=['nom_estacio', 'data', 'contaminant', 'unitats', 'latitud', 'longitud', 'no2'])
print(df.head())
