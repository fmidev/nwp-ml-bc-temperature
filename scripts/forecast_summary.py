import pandas as pd
from pathlib import Path 
import os
import re

# Set the number of CPUs used
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

# Path to data 
MY_DATA_DIR = Path.home() / "thesis_project" / "data" / "parquet_data"

# Output directory
OUT = Path.home() / "thesis_project" / "data" 

# Separate the weather parameters 
WEATHER_VARS = ["MSL","T2","D2","U10","V10","LCC","MCC","SKT","MX2T","MN2T","T_925","T2_ENSMEAN_MA1","T2_M1","T_925_M1","SSR_Acc","STR_Acc"]


def agg_summary(filename_pattern):

    """
    A function to create a summary dataframe aggregated over the stations with results by month.
    Summary includes nans and, min, max and med values for the numeric fields, for each variable
    Params: 
        filename_patter = Pattern of the filenames to be read for the summary
    Returns: A dataframe of the summary  
    """
    out_rows = []

    # Iterate over all of the parquet files and calculate summary 
    for file in MY_DATA_DIR.glob(filename_pattern):

        # Read the file into data frame
        df = pd.read_parquet(file)
        m = re.search(r"part-(\d{4})-(\d{1,2})", file.name)
        year, month = m.groups()

        # Create a summary data frame with a row for each parameter 
        # The summary includes values on Nan values, min, max and median numbers 
        # These values are taken across all stations per month (so it only reports singular minimum value from any of the stations...)
        summary = pd.DataFrame({
            "n_rows" : len(df),
            "nan_count" : df.isna().sum(),
            "nan_frac" : df.isna().mean(),
            "min" : df.min(numeric_only = True),
            "max" : df.max(numeric_only = True),
            "median" : df.median(numeric_only = True),})
        
        # Append the summary data frame 
        summary["year"] = year
        summary["month"] = month
        out_rows.append(summary.reset_index(names = "column"))

        # Print the file name to keep track of how many files are done
        print(file.name)

    # Concatenate all of the montly summaries to get the summary across the full time scale
    final_summary = pd.concat(out_rows, ignore_index = True)

    return final_summary



def outliers(filename_pattern, nan_frac, exc_nan, valid_ranges):
    """
    A function to create a summary dataframe of outlying values. 
    Summary includes two types of outlying types: unrealistic values for the numeric weather parameters and a threshold check for large portion of NAN values
    Checks values for each station and flags stations that have unrealistic predicted weather values or many NAN values for some parameter/parameters
    Params: 
        filename_patter = Pattern of the filenames to be read for the summary
        nan_frac = Threshold for the amount on nan values (if over the station gets flagged)
        exc_nan = Some variables are known to have a large number of NAN values due to how/when they are calculated so that the stations don't
                    get flagged by these 
        valid_ranges = Dictionary of realistic values for the weather parameters 
    Returns: A dataframe of the flagged stations and the values that caused the flag
    """

    flags = []

    # Go through all of the files 
    for file in MY_DATA_DIR.glob(filename_pattern):
        df = pd.read_parquet(file)

        # Extract the month and year from the filename for later 
        m = re.search(r"part-(\d{4})-(\d{1,2})", file.name)
        year, month = m.groups()

        # Group by the station id to get station specific flags
        for sid, group in df.groupby("SID"):
            # Check that all of the weather parameter columns are present
            cols_present = [c for c in WEATHER_VARS if c in group.columns]

            # Flag unreasonable values
            for var, (low, high) in valid_ranges.items():
                if var in group.columns:
                    bad = group[(group[var] < low) | (group[var] > high)]
                    if not bad.empty:
                        flags.append({
                            "flag_type": "unrealistic_vals",
                            "year": year,
                            "month": month,
                            "SID": sid,
                            "column": var,
                            "n_bad": len(bad),
                            "min_bad": bad[var].min(),
                            "max_bad": bad[var].max(),
                        })
            
            

            # Drop parameters from Nan check 
            nan_cols = [c for c in cols_present if c not in exc_nan]

            # Count Nan values
            nan_counts = group[nan_cols].isna().sum()
            nan_fracs = group[nan_cols].isna().mean()
            n_total = len(group)

            # Flag unreasonably high number of nan values
            for var in nan_cols:
                thr = nan_frac
                frac = round(float(nan_fracs[var]), 2)
                if frac >= thr:
                    flags.append({
                        "flag_type": "nan_vals",
                        "year": year,
                        "month": month,
                        "SID": sid,
                        "column": var,
                        "nan_count": int(nan_counts[var]),
                        "nan_frac": frac,
                        "nan_frac_threshold": thr,
                        "n_total_rows": int(n_total),
                    })

        # Print the file name to check process       
        print(file.name)


    # Save flagged stations/columns
    flags_df = pd.DataFrame(flags)

    return flags_df

def main():

    # Filename pattern 
    filename_pattern = "*.parquet"
    # Set to later exclude from nan value flagging because the parameter only calculated for few timepoints  
    exc_nan = {"T2_ENSMEAN_MA1"}

    # Set fraction for nan values to get flagged 
    nan_frac= 0.30

    # Set up plausible value ranges for the weather parameters
    valid_ranges = {
        "T2": (-80+273.15, 60+273.15),                    # K    
        "D2": (-80+273.15, 60+273.15),                    # K 
        "U10": (-100, 100),                               # m/s
        "V10": (-100, 100),                               # m/s
        "MSL": (80000, 110000),                           # Pa
        "LCC": (0, 1),                                    # fraction
        "MCC": (0, 1),                                    # fraction
        "SKT": (-80+273.15, 70+273.15),                   # K
        "MX2T": (-80+273.15, 70+273.15),                  # K
        "MN2T": (-80+273.15, 70+273.15),                  # K
        "T_925": (-80+273.15, 60+273.15),                 # K 
        "T2_M1": (-80+273.15, 60+273.15),                 # K 
        "T_925_M1": (-80+273.15, 60+273.15),              # K
        "T2_ENSMEAN_MA1": (-80+273.15, 60+273.15),        # K 
        "SSR_Acc": (0, 1500),
        "STR_Acc": (-500, 50)
    }


    # Save the summary as a csv file
    final_summary = agg_summary(filename_pattern)
    final_summary.to_csv(OUT / "summary_all_forecasts.csv", index = False)
    print("Summary done")

    flags_df = outliers(filename_pattern, nan_frac, exc_nan, valid_ranges)
    flags_df.to_csv(OUT / "flagged_outliers.csv", index=False)
    print("Flags done")

if __name__ == "__main__":
    main() 