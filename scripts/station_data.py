import pandas as pd 
from pathlib import Path

# Read the station data into a pandas data frame 
stations = pd.read_csv("/data/MOS_data/mos_stations_limited_Europe.csv")

# Remove unnecessary column producer
stations = stations.drop(columns=["producer"])

MY_DATA_DIR = Path.home() / "thesis_project" / "data" 


output_file = MY_DATA_DIR / "stations.csv"

# Save the cleaned data to a csv file
stations.to_csv(output_file, index=False)