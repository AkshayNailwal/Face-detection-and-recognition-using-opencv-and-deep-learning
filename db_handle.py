import psycopg2

connection = psycopg2.connect(host="", db_name="", usr_name="", paswd="")
curs = connection.cursor()

def create_Database(curs):
    query = "CREATE DATABASE Face_Recognition IF NOT EXISTS"
    curs.execute(query)