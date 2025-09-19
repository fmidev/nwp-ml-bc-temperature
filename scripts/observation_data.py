import pandas as pd 
from pathlib import Path
import glob 
import os


csv_folder = "/data/MOS_data/observations/"
filename_pattern = "*.csv"

MY_DATA_DIR = Path.home() / "thesis_project" / "data"
output_file = MY_DATA_DIR / "observations.csv"


all_csvs = sorted(glob.glob(os.path.join(csv_folder, filename_pattern)))

full_observations = []
for csv_file in all_csvs:
    print(f"Processing {csv_file}")
    
    # Read CSV
    df = pd.read_csv(csv_file)
    full_observations.append(df)

observations = pd.concat(full_observations)
print(observations.head())

# Save the cleaned data to a csv file
observations.to_csv(output_file, index=False)
print("CSV file saved")



