import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Path to data folder
csv_folder = "/data/MOS_data/forecasts"  

# Output folder
MY_DATA_DIR = Path.home() / "thesis_project" / "data" / "parquet_data"

# Pattern for the existing filenames
filename_pattern = "*.csv"   


def parse_year_month(filename: str):
    """
    Extract year and month from filename.
    """
    basename = os.path.basename(filename)
    digits = ''.join([c for c in basename if c.isdigit()])
    if len(digits) >= 6:
        year, month = digits[:4], digits[4:6]
        return int(year), int(month)
    else:
        raise ValueError(f"Could not parse year/month from {basename}")

def main():
    os.makedirs(MY_DATA_DIR, exist_ok=True)
    all_csvs = sorted(glob.glob(os.path.join(csv_folder, filename_pattern)))

    for csv_file in all_csvs:
        year, month = parse_year_month(csv_file)
        print(f"Processing {csv_file} -> year={year}, month={month}")

        # Read CSV
        df = pd.read_csv(csv_file)

        # Convert to Arrow Table
        table = pa.Table.from_pandas(df)

        # Partitioned write: each year/month will get its own folder
        # Save as parquet to save memory 
        pq.write_to_dataset(
            table,
            root_path=MY_DATA_DIR,
            partition_cols=["year", "month"] if "year" in df.columns and "month" in df.columns else None,
            basename_template=f"part-{year}-{month}-{{i}}.parquet"
        )

if __name__ == "__main__":
    main()