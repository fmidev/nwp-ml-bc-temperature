"""
Script to parse together the forecasts, observations and geographical information for the stations
Saves the merged data frames as parquet files, such that there are one file per month
"""
import pandas as pd
from pathlib import Path 
import os
from collections import defaultdict
import re



os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

# Paths
MY_DATA_DIR_PAR = Path.home() / "thesis_project" / "data" / "parquet_data"
MY_DATA_DIR = Path.home() / "thesis_project" / "data"
OUT             = Path.home() / "thesis_project" / "data" / "combined_parquet"

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
    m = re.match(r"part-(\d{4})-(\d{1,2})-", pth.name)

    # Check that the parsing is possible
    if not m:
        raise ValueError(f"Cannot parse month from {pth.name}")
    
    # Return the year and month
    return f"{m.group(1)}-{int(m.group(2)):02d}"


def key_formatting(df, keys):

    """
    Helper function to format the column values of the join keys
    Param: 
        df = Dataframe to be formatted
        keys = list of columns names to format
    Returns: Formatted data frame
    """

    # Iterate through the keys
    cols = [c for c in keys if c in df.columns]

    # Set the columns as type string
    df[cols] = df[cols].astype("string[pyarrow]")

    # Format the columns
    for c in cols:
        df[c] = df[c].str.strip()

    return df



def main():
    # Filenames/namepatterns
    filename_pattern = "*.parquet"
    filename_obs = "observations.csv"
    filename_stations = "stations.csv"

    # Read the station information from the file 
    df_stations = pd.read_csv(MY_DATA_DIR / filename_stations)

    # Format the column values of the join key
    df_stations = key_formatting(df_stations, ["SID"])


    # Read the observations from the file 
    df_obs = pd.read_csv(MY_DATA_DIR / filename_obs)

    # Format the column values of the join keys 
    df_obs = key_formatting(df_obs, ["SID", "obstime"])

    # Get the month of the observation from the observation time to help with matching
    df_obs["obs_month"] = df_obs["obstime"].str.slice(0, 7)


    # Slice observations by month (rename obsrvation time to validtime to make the later joining easier)
    obs_by_month = {
        m: g.rename(columns={"obstime": "validtime"})[["SID", "validtime", "obs_TA"]].copy()
        for m, g in df_obs.groupby("obs_month", sort=True)
    }

    # Create buckets to help with writing the files
    buckets = defaultdict(list)

    # Run through the monthly forecast files 
    for file in MY_DATA_DIR_PAR.glob(filename_pattern):

        # Load the parquet file into a data frame 
        df_par = pd.read_parquet(file)

        # Format the column values of the join keys
        df_par = key_formatting(df_par, ["SID", "analysistime", "validtime"])


        df_par = df_par.merge(
            df_stations,
            on="SID",
            how="left",
            validate="m:1"  
        )

        # Use the month of the file as the bucket key
        month_key = ym_from_filename(file)

        # Get the month of the forecasts 
        vmonths = df_par["validtime"].str[:7].unique().tolist()

        # Get the matching observation slice (month)
        obs_parts = [obs_by_month.get(m) for m in vmonths]
        obs_parts = [x for x in obs_parts if x is not None]
        obs_m = (pd.concat(obs_parts, ignore_index=True)
                if obs_parts else pd.DataFrame(columns=["SID","validtime","obs_TA"]))
        


        # Merge the observations to the forecast based on station ID and the validtime(forecast) and observationtime(observation) 
        combined = df_par.merge(
            obs_m, on=["SID","validtime"], how="left", validate="m:1"
        )


    
        # Appendt the merged data frame into the bucket
        buckets[month_key].append(combined)

        # Print the bucket, month and the lenght of the data frame to follow the progress 
        print(f"bucketed file={file.name} → month={month_key}, rows={len(combined):,}")

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
        out_path = OUT / f"combined_{month}.parquet"
        final_df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} | rows: {len(final_df):,}")

if __name__ == "__main__":
    main() 


