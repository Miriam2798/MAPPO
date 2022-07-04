import psycopg2
from psycopg2 import Error
 
def data():
    try:
        # Connect to an existing database
        connection = psycopg2.connect(user="feflopfeklpznc",
                                    password="5de7e5b5fc9f83e323359f1c4ba05394ed23356dd8ba561aa45b88b54d11c026",
                                    host="ec2-54-73-167-224.eu-west-1.compute.amazonaws.com",
                                    port="5432",
                                    database="dfi7i5f0k4mkd2")
 
        # Create a cursor to perform database operations
        cursor = connection.cursor()
        # Print PostgreSQL details
        print("PostgreSQL server information")
        print(connection.get_dsn_parameters(), "\n")
        # Executing a SQL query
        #cursor.execute("select * from public.ranking")
        cursor.execute("""
                        SELECT nom_estacio, "data", contaminant, unitats, latitud, longitud, no2
                        FROM definitiu.no2_csv

                        """)
       
        # Fetch result
        record = cursor.fetchall()
        return record
 
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
