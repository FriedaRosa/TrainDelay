from deutsche_bahn_api import *
from train_delay import *
from dotenv import load_dotenv
import os

import pandas as pd
import mysql.connector


load_dotenv()  # Load variables from .env

auth_data = AuthData("c69bc083818a8db0a5273d6f1c4c53b7", "b7522d5270140afc580ed546b9e6da98")
database_config = DatabaseConfig(os.getenv("DB_HOST"), os.getenv("DB_USER"), os.getenv("DB_PASSWORD"), os.getenv("DB_NAME"))

train_delay_tracker = TrainDelayTracker(auth_data, database_config)

train_delay_tracker.track_station("Köln Hbf")

# on your local machine
connection = mysql.connector.connect(
    host= os.getenv("DB_HOST"),
    user= os.getenv("DB_USER"),
    password= os.getenv("DB_PASSWORD"),
    database= os.getenv("DB_NAME")
    )

df = pd.read_sql("SELECT * FROM train_delay_status", connection)
df.to_csv("data/train_data.csv", index=False)
connection.close()