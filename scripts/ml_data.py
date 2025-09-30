import pandas as pd
from pathlib import Path 
import os
from collections import defaultdict
import re
import numpy as np


os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

# Paths
MY_DATA_DIR = Path.home() / "thesis_project" / "data" / "combined_parquet"
OUT = Path.home() / "thesis_project" / "data" / "ml_data"
OUT.mkdir(parents=True, exist_ok=True)

def ym_from_filename(pth):
    """ 
    Helper function to parse the year and month from the file name
    Params:
        Path to the data
    Returns The year and the month
    """
    # Strip the year and month from the file
    # Accept 1- or 2-digit month in the file name 
    m = re.match(r"combined_(\d{4})-(\d{1,2})", pth.name)

    # Check that the parsing is possible
    if not m:
        raise ValueError(f"Cannot parse month from {pth.name}")
    
    # Return the year and month
    return f"{m.group(1)}-{int(m.group(2)):02d}"

def main():
    # Filename pattern
    filename_pattern = "*.parquet"

    # Create buckets to help with writing the files
    buckets = defaultdict(list)

    # Run through the monthly forecast files 
    for file in MY_DATA_DIR.glob(filename_pattern):

        # Load the parquet file into a data frame 
        df = pd.read_parquet(file)

        month_key = ym_from_filename(file)

        # Drop the radiation parameters 
        df = df.drop(columns=["SSR_Acc", "STR_Acc"])

        # ENSMEAN to NAN
        col = "T2_ENSMEAN_MA1"

        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].where(df[col] > 100)


        # Sin/Cos times for hod and doy
        df["ts"] = pd.to_datetime(df["validtime"], errors="coerce")

        #  Hour-of-day (sin/cos)
        # fractional hour in [0, 24)
        h = df["ts"].dt.hour + df["ts"].dt.minute/60 + df["ts"].dt.second/3600
        theta_h = 2 * np.pi * (h / 24.0)
        df["sin_hod"] = np.sin(theta_h)
        df["cos_hod"] = np.cos(theta_h)

        #  Day-of-year (sin/cos), leap-year aware
        # Use fractional day-of-year to avoid a jump at midnight
        doy = df["ts"].dt.dayofyear.astype("Float64")          # 1..365/366
        sec = (df["ts"].dt.hour*3600 + df["ts"].dt.minute*60 + df["ts"].dt.second).astype("Float64")
        frac_doy = (doy - 1) + (sec / 86400.0)                 # 0..(365/366)

        # Year length: 366 for leap years else 365
        year_len = df["ts"].dt.is_leap_year.map({True: 366.0, False: 365.0}).astype("Float64")

        theta_y = 2 * np.pi * (frac_doy / year_len)
        df["sin_doy"] = np.sin(theta_y)
        df["cos_doy"] = np.cos(theta_y)

        # Remove closed stations and stations that opened late
        df = df[~df["closed"]]
        df = df[~df["opened_late"]]

        # Filter to only stations that send observations hourly
        df = df[df["tag"] == "1h"]

        # Remove duplicate rows
        df_unique = df.drop_duplicates()
        rm = len(df) - len(df_unique)
        print(f"\n Duplicate rows removed {rm}")
    
        # Appendt the merged data frame into the bucket
        buckets[month_key].append(df_unique)

        # Print the bucket, month and the lenght of the data frame to follow the progress 
        print(f"bucketed file={file.name}")

    # Go through parts for writing the combined files
    for month, parts in buckets.items():

        final_df = pd.concat(parts, ignore_index=True)
        

        # Sort the data frame rows by the lead time
        final_df["_lt"] = pd.to_numeric(final_df["leadtime"], errors="coerce")
        final_df = final_df.sort_values(
            ["SID","analysistime","_lt","validtime"],
            ascending=[True, True, True, True],
            kind="stable"
        ).drop(columns="_lt").reset_index(drop=True)


        if "analysishour" in final_df.columns:
            final_df["analysishour"] = final_df["analysistime"].str[11:13]

        # Write the combined files 
        out_path = OUT / f"ml_data_{month}.parquet"
        final_df.to_parquet(out_path, index=False)
        print(final_df["SID"].nunique())
        print(final_df.memory_usage().sum())
        #print(final_df.head(3))
        print(f"Wrote {out_path} | rows: {len(final_df):,}")

if __name__ == "__main__":
    main() 
