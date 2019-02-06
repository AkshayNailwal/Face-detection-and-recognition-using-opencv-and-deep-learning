import mysql.connector
import config.DATABASE


def create_Database(curs):
    query = "CREATE DATABASE {} IF NOT EXISTS".format(db_name)
    curs.execute(query)