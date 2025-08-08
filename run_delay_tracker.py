from deutsche_bahn_api import *
from train_delay import *

auth_data = AuthData("c69bc083818a8db0a5273d6f1c4c53b7", "b7522d5270140afc580ed546b9e6da98")
database_config = DatabaseConfig("localhost", "root", "igaW1997!!", "delayedtrains")

train_delay_tracker = TrainDelayTracker(auth_data, database_config)
train_delay_tracker.get_station_data("Köln Hbf") 

train_delay_tracker.track_station("Köln Hbf")
train_delay_tracker.track_station("Spich")
