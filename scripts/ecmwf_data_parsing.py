import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Output folder
OUT = Path.home() / "thesis_project" / "data" / "parquet_data"

# Make output directory
os.makedirs(OUT, exist_ok=True)

# Path to data folder
CSV_FOLDER = "/data/MOS_data/forecasts"


def parse_year_month(filename):
    """ 
    Helper function to parse the year and month from the file name
    Params:
        filename = name of the file 
    Returns The year and the month
    """
    # Get the base format of the filename
    basename = os.path.basename(filename)

    # Get the digits of the filename
    digits = ''.join([c for c in basename if c.isdigit()])

    # Check the length
    if len(digits) >= 6:
        # Extract the year and month 
        year, month = digits[:4], digits[4:6]
        # Return year and month
        return int(year), int(month)
    else:
        raise ValueError(f"Could not parse year/month from {basename}")

def main():
    # Pattern for the existing filenames
    filename_pattern = "*.csv"

    # List for original file lengths
    original_rows = [] 

    # Get all of the csv file
    all_csvs = sorted(glob.glob(os.path.join(CSV_FOLDER, filename_pattern)))

    # Go through all of the original csv files
    for csv_file in all_csvs:

        # Extract the year and month from the filename
        year, month = parse_year_month(csv_file)

        # Print to monitor progress
        print(f"Processing {csv_file} -> year={year}, month={month}")

        # Read CSV
        df = pd.read_csv(csv_file)

        # Get the number of rows
        n_rows = len(df)
            

        # Record the original number of rows for each monthly file
        original_rows.append({
            "file": os.path.basename(csv_file),
            "year": year,
            "month": month,
            "rows_csv": n_rows
        })

        # Convert to Arrow Table
        table = pa.Table.from_pandas(df)

        # Partitioned write: each year/month will get its own file
        # Save as parquet to save memory 
        pq.write_to_dataset(
            table,
            root_path=OUT,
            partition_cols=["year", "month"] if "year" in df.columns and "month" in df.columns else None,
            basename_template=f"part-{year}-{month}-{{i}}.parquet"
        )
        
    # Save the csv with the original file lengths
    original_df = pd.DataFrame(original_rows).sort_values(["year", "month", "file"])
    out_csv = OUT / "csv_rowcounts_by_file.csv"
    original_df.to_csv(out_csv, index=False)
    print(f"Wrote per-file counts to {out_csv}")

if __name__ == "__main__":
    main() 
