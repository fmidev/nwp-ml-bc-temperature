import pandas as pd 
from pathlib import Path
import glob 


files = glob.glob("/data/MOS_data/observations/*.csv")

observations = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

MY_DATA_DIR = Path.home() / "thesis_project" / "data"

output_file = MY_DATA_DIR / "observations.csv"

# Save the cleaned data to a csv file
observations.to_csv(output_file, index=False)