"""
Script to check the parsed data to see if the parsing was done succesfully
"""

import re
import pandas as pd
from pathlib import Path
import os
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

# Paths 
COMBINED_DIR = Path.home() / "thesis_project" / "data" / "combined_parquet"
FORECAST_DIR = Path.home() / "thesis_project" / "data" / "parquet_data"
OBSERVATIONS = Path.home() / "thesis_project" / "data" / "observations.csv"
ML_DIR = Path.home() / "thesis_project" / "data" / "ml_data"
STATIONS = Path.home() / "thesis_project" / "data" / "stations.csv"
CSV = FORECAST_DIR / "csv_rowcounts_by_file.csv"   




def ym_from_combined(p):

    """ 
    Helper function to parse the year and month from the merged parquet file name
    Params:
        Path to the data
    Returns The year and the month
    """

    # Strip the year and month from the file name
    ym = re.match(r"combined_(\d{4}-\d{2})\.parquet$", p.name)

    # Return the year and month
    return ym.group(1) if ym else None


def ym_from_forecast(p):

    """ 
    Helper function to parse the year and month from the forecast parquet file name
    Params:
        Path to the data
    Returns The year and the month
    """
    
    # Strip the year and month from the file name
    # Accept 1- or 2-digit month in the file name
    ym = re.match(r"part-(\d{4})-(\d{1,2})-", p.name)

    # Return the year and month
    #year, mon = ym.group(1), int(ym.group(2))
    #return f"{year}-{mon:02d}" if ym else None
    return f"{ym.group(1)}-{int(ym.group(2)):02d}"

def check_file_length_match():

    """ 
    This function loads the information of the forecast data, the combined data and some infromation from the original data to check that the 
    lengths of all of the files (original forecast, forecast in parquet format and combined forecast + observations) are equal. This is done in order
    to check that the parsing went correctly and no rows were lost or added or went into the wrong split (as the data is split by month)
    Params: 

    Returns: List of matches/mismatches and lists for the file lenghts 
    """
        
    # Combined files
    combined_files = sorted(COMBINED_DIR.glob("combined_*.parquet"))

    # Year and month from combined file
    ym_c = [ym_from_combined(p) for p in combined_files]

    # Length of the files
    len_com = [len(pd.read_parquet(p)) for p in combined_files]


    # Forecast files
    forecast_files = list(FORECAST_DIR.glob("part-*.parquet"))

    # Year and month from forecast file 
    ym_f = [ym_from_forecast(i) for i in forecast_files]

    # Length of the files
    len_forecast = [len(pd.read_parquet(i)) for i in forecast_files]


    # Initialize the map variable
    csv_map = {}

    # Check that the CSV file exists 
    if CSV.exists():

        # Read the file 
        csv_df = pd.read_csv(CSV) 

        # Generate year-month date (YYYY-MM) to match the other files
        csv_df["month_str"] = csv_df.apply(
            lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}", axis=1
        )
        # Iterate over the rows of the CSV map the corresponding lenghts and dates (year-month)
        for _, r in csv_df.iterrows():
            csv_map[r["month_str"]] = int(r["rows_csv"])
        
    else:
        print(f"WARNING: {CSV} not found; CSV counts will be None.")

    # List the lenghts from the mapping
    len_csv = [csv_map.get(m) for m in ym_c]

    # Print the leghts of the combined files, forecast files, and the original data files 
    # Also print match or mismatch depending on if the lenghts are equal
    status_list = []
    length_combined = []
    length_forecast = []
    length_csv = []
    for i, (m, a, b, c) in enumerate(zip(ym_c, len_com, len_forecast, len_csv)):
        status = "MATCH" if (a == b == c) else "MISMATCH"
        status_list.append(status)
        length_forecast.append(b)
        length_combined.append(a)
        length_csv.append(c)

    return status_list, length_combined, length_forecast, length_csv



def check_temp_matches(sid, timepoint):

    """ 
    This function loads the information of the observations and the combined data to check that the timepoints have the correct observed temperature
    This is done for one station to limit the data amount. It also prints the rows of the combined data frame for one time point to check the temperatures
    because there are multiple predictions for one time point but these should all have the same observed temperature.
    Params: 
        sid = Unique station id
        timepoint = One date and time
    Returns: Data frame of mismatched rows and data frame for one singular timepoint
    """

    # Load the observations 
    observations = pd.read_csv(OBSERVATIONS)
    observations["SID"] = observations["SID"].astype("string")

    # Filter to only one station to make processing faster
    observed_one = observations[observations["SID"] == sid].copy()

    # Change the obstime to validtime to make merging more simple
    observed_one = observed_one.rename(columns = {"obstime":"validtime"}) 



    forecasts = []
    filename_pattern = "*.parquet"

    # Loop throught the combined files
    for file in COMBINED_DIR.glob(filename_pattern):

        # Read the file 
        df = pd.read_parquet(file)

        # Filter to the same station as with the observations
        df_one = df[df["SID"] == sid]

        # Change None values to NaN 
        df_one = df_one.fillna(value=np.nan)

        # Append to a list to get all values from the 10 year period 
        forecasts.append(df_one)


    # Concatenate the data frames 
    forecasts_one = pd.concat(forecasts, ignore_index=True)

    # Merge the two data frames for the same station
    merged = forecasts_one.merge(observed_one[['SID', 'validtime', 'obs_TA']], 
                    on=['SID', 'validtime'], 
                    suffixes=('_df1', '_df2'), 
                    how='inner')

    # Check where obs_TA  matches to check that the obs_TA came to the correct timepoint 
    # Also count NaN and NaN as a match 
    merged['obs_TA_match'] = (
        merged['obs_TA_df1'].eq(merged['obs_TA_df2']) | 
        (merged['obs_TA_df1'].isna() & merged['obs_TA_df2'].isna())
    )

    one_timepoint = merged[merged["validtime"] == timepoint]

    # Get rows with mismatches
    mismatches = merged[~merged['obs_TA_match']]

    return mismatches, one_timepoint

def check_station_matches(sid, timepoint):

    """ 
    This function loads the information of the stations and the combined data to check that the corresponding station has the correct geographical information
    This is done for one station to limit the data amount. It also prints the rows of the combined data frame for one time point to check the information
    that the same station ID always has the same information
    Params: 
        sid = Unique station id
        timepoint = One date and time
    Returns: Data frames of mismatched rows for latitude, longitude, and elevation and data frame for one singular timepoint
    """

    # Load the observations 
    stations = pd.read_csv(STATIONS)
    stations["SID"] = stations["SID"].astype("string")

    # Filter to only one station to make processing faster
    stations_one = stations[stations["SID"] == sid].copy()

    forecasts = []
    filename_pattern = "*.parquet"

    # Loop throught the combined files
    for file in COMBINED_DIR.glob(filename_pattern):

        # Read the file 
        df = pd.read_parquet(file)

        # Filter to the same station as with the observations
        df_one = df[df["SID"] == sid]

        # Change None values to NaN 
        df_one = df_one.fillna(value=np.nan)

        # Append to a list to get all values from the 10 year period 
        forecasts.append(df_one)


    # Concatenate the data frames 
    forecasts_one = pd.concat(forecasts, ignore_index=True)


    # Merge the two data frames for the same station
    merged = forecasts_one.merge(stations_one, 
                    on=['SID'], 
                    suffixes=('_df1', '_df2'), 
                    how='inner')

    # Check where obs_TA  matches to check that the obs_TA came to the correct timepoint 
    # Also count NaN and NaN as a match 
    merged['lat_match'] = (
        merged['lat_df1'].eq(merged['lat_df2'])
    )
    merged['lon_match'] = (
        merged['lon_df1'].eq(merged['lon_df2'])
    )
    merged['elev_match'] = (
        merged['elev_df1'].eq(merged['elev_df2'])
    )

    # Get rows with mismatches
    mismatches_lat = merged[~merged['lat_match']]
    mismatches_lon = merged[~merged['lon_match']]
    mismatches_elev = merged[~merged['elev_match']]

    one_timepoint = merged[merged["validtime"] == timepoint]

    return mismatches_lat, mismatches_lon, mismatches_elev, one_timepoint 


def main():

    
    # Get the matches and file lengths from the files
    match, comb, forecast, csv = check_file_length_match()
    #print(match)
    
    # If there is a mismatch in the lengths print the lengths
    for i in range(len(match)):
        if match[i] == "MISMATCH":
            print(f"Length of combined file {comb[i]}")
            print(f"Length of forecast file {forecast[i]}")
            print(f"Length of original CSV file {csv[i]}")
    
    # Set a specific station ID and timepoint
    sid = "100859"
    timepoint = "2015-10-06 03:00:00"

    # Get mismatched rows of temperature observations and rows of one timepoint
    mismatched, one_time = check_temp_matches(sid, timepoint)

    # Print results
    print("Mismatched temperatures")
    print(mismatched)
    print("One timepoint")
    print(one_time)

    
    # Get mismatched rows of geographical parameters nad and rows of one timepoint
    lat, lon, elev, one_time = check_station_matches(sid, timepoint)

    # Print results 
    print("Latitude mismatches")
    print(lat)
    print("Longitude mismatches")
    print(lon)
    print("Elevation mismathces")
    print(elev)
    print("One timepoint")
    print(one_time)
    

if __name__ == "__main__":
    main() 
    







