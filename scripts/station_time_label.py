# Tag station with the time resolution and check if the have opened late or closed
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Sequence, Tuple

# Paths
OBS_FILE = Path.home() / "thesis_project" / "data" / "observations.csv"
STATION_FILE = Path.home() / "thesis_project" / "data" / "stations.csv"
OUT = Path.home() / "thesis_project" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def classify_station_from_grid(df, time_col="obstime", value_col="obs_TA", rolling_days=30):
    """
    Classify station cadence using presence/absence patterns on an hourly grid:
      - '1h'      : Mostly filled
      - '3h'      : ~ One of every 3 hours filled
      - '3h→1h'   : Early ~3h, later ~1h
      - 'mixed'   : None of the above
      - 'empty'   : No valid observations
        Params: 
            df = Dataframe of stations and observations 
            time_col = Name of the observation time column
            value_col = Name of the observation value column
            rolling_days = Number of rolling days
        Return: Classification tag for the station 
    """

    # If no observations return empty
    if df.empty:
        return "empty"

    # Create a copy of the dataframe
    g = df.copy()

    # Change the time to datetime type and drop NaN values
    g[time_col] = pd.to_datetime(g[time_col], utc=True, errors="coerce")
    g = g.dropna(subset=[time_col]).sort_values(time_col)

    # Use time as index (required for rolling('30D'))
    g = g.set_index(time_col)

    # Boolean; whether this hour has a valid observation
    valid = g[value_col].notna()

    # If no valid observations return empty
    if valid.empty:
        return "empty"

    # Time-based rolling fill-rate over the past N days
    win = f"{rolling_days}D"

    # Fraction of hours with data in last N days
    obs_rate = valid.rolling(win).mean()  

    # Heuristics 
    one_like   = obs_rate > 0.70                            # mostly hourly
    three_like = (obs_rate >= 0.25) & (obs_rate <= 0.45)    # ~1 out of 3 hours

    # If we have too few timestamps to judge, fall back to overall fill rate
    if obs_rate.notna().sum() < 24 * 7:  # <1 week of signal
        overall = valid.mean()
        if overall > 0.70:   return "1h"
        if 0.25 <= overall <= 0.45: return "3h"
        return "mixed"

    # Mostly one regime
    if one_like.mean() > 0.80:
        return "1h"
    if three_like.mean() > 0.80:
        return "3h"

    # Detect switch; early mostly 3h, late mostly 1h
    n = len(obs_rate)
    early = three_like.iloc[: n//2].mean()
    late  = one_like.iloc[n//2 :].mean()
    if early > 0.60 and late > 0.60:
        return "3h→1h"

    return "mixed"


def closure_check(df, reference_date, time_col = "obstime", value_col = "obs_TA", gap_days = 30):
    """
    Function to check if the station has been closed with the following conditions
    Strict closure:
      closed == True if
        - last_obs exists, and
        - (ref_date - last_obs) >= gap_days,
        - None of the rows after last_obs have valid values in value_col.
    Params: 
        df = Dataframe with station information and observations
        reference_date = The reference date to compare against (last day of values from any station)
        value_col = Name of the observation colums 
        gap_days = Days required to not have observations for closure check
    Returns: 
        Boolean: Closed/Not Closed
        Date of closure
    """
    
    # Check that dataframe is not empty
    if df is None or df.empty:
        # No rows at all -> treat as not enough info to call closed.
        return False, pd.NaT

    # Create a copy of the dataframe
    g = df.copy()

    # Change the observation time to datetime type and drop NaN values
    g[time_col] = pd.to_datetime(g[time_col], utc=True, errors="coerce")
    g = g.dropna(subset=[time_col]).sort_values(time_col)

    # Check that there are timepoints
    if g.empty:
        return False, pd.NaT

    # Get valid observations
    any_valid = g[value_col].notna()
    
    # Check that there are valid observations
    if any_valid.empty:
        return False, pd.NaT


    # Reference date
    ref_date = reference_date

    # Last timestamp with any valid value
    last_obs = g.loc[any_valid, time_col].max() if any_valid.any() else pd.NaT

    # Rows strictly after the last valid observation
    later = g[g[time_col] > last_obs]

    # Check for any valid observations after 
    any_valid_after = later[value_col].notna().any()

    # Gap relative to reference date
    gap = (pd.Timestamp(ref_date) - last_obs).days
    long_enough_gap = (gap is not None) and (gap >= gap_days)

    # Closed based on the requirements
    closed = (long_enough_gap) and (not any_valid_after)
    
    # Date of the last observation
    closed_since = last_obs if closed else pd.NaT

    return bool(closed), closed_since

def opening_check(df, reference_date, time_col = "obstime", value_col = "obs_TA", lead_gap_days: int = 30, min_valid_after_rows: int = 24):
    """
    Function to check if a station has been opened after the start of the data period with the following conditions
        Strict opening:
        opened_late == True if
        - the span from first timestamp to first valid observation >= lead_gap_days,
        - and there are at least `min_valid_after_rows` valid rows after first valid.
        Params:
            df = Dataframe with station information and observations
            reference_date = Reference date to compare against (first day of values from any station)
            value_col = Name of the observation colums  
            lead_gap_days = Number of days of 'no data' required before first valid observation to be considered opened late
            min_valid_after_rows = Number of rows required after the first valid obsevations to check for sustained observations 
        Returns: 
            Boolean Opened late/Not opened late
            Date of opening
    """
    # Check if the dataframe is empty
    if df is None or df.empty:
        return False, pd.NaT

    # Make a copy of the dataframe
    g = df.copy()

    # Change the time to datetime type and drop NaN values
    g[time_col] = pd.to_datetime(g[time_col], utc=True, errors="coerce")
    g = g.dropna(subset=[time_col]).sort_values(time_col)

    # Check that there are timepoints
    if g.empty:
        return False, pd.NaT

    # Get valid observations
    any_valid = g[value_col].notna()

    # Check that there are valid observations
    if any_valid.empty:
        return False, pd.NaT

    # Reference date
    ref_date = pd.Timestamp(reference_date)

    # First time we actually had any value
    first_obs = g.loc[any_valid, time_col].min() if any_valid.any() else pd.NaT

    # Duration of value-less period before first valid
    lead_gap = (first_obs - ref_date).days
    long_enough_lead_gap = (lead_gap is not None) and (lead_gap >= lead_gap_days)

    # After first valid: ensure it's not a single blip
    later = g[g[time_col] >= first_obs]  # include the first valid row
    valid_after_cnt = later[value_col].notna().sum()

    # Check that there are sustained observations after
    has_sustained_after = valid_after_cnt >= min_valid_after_rows

    # Mark station as opened late if requirments are met
    opened_late = bool(long_enough_lead_gap and has_sustained_after)

    # Date of opening
    opened_since = first_obs if opened_late else pd.NaT

    return opened_late, opened_since


def main():

    # Read station information into a dataframe
    stations = pd.read_csv(STATION_FILE)

    # Read observations into a dataframe
    obs = pd.read_csv(OBS_FILE)

    # Parse time once (UTC) and drop obviously bad times
    obs["obstime"] = pd.to_datetime(obs["obstime"], utc=True, errors="coerce")
    obs = obs.dropna(subset=["obstime"])

    # Global bounds across ALL stations
    GLOBAL_START = obs["obstime"].min()
    GLOBAL_END   = obs["obstime"].max()

    # Run through all of the station IDs and group observations by station ID
    results = []
    for sid, group in obs.groupby("SID"):

        # Get tag for the station 
        tag = classify_station_from_grid(group, time_col="obstime", value_col="obs_TA")

        # Check if the station has been closed
        closed, closed_since = closure_check(group, reference_date=GLOBAL_END, time_col="obstime", value_col="obs_TA", gap_days=30)

        # Check if station was opened late
        opened, opened_since = opening_check(group, reference_date=GLOBAL_START, time_col="obstime", value_col="obs_TA", lead_gap_days=30, min_valid_after_rows=24)
            
        # Append the results for the station 
        results.append({
            "SID": sid,
            "tag": tag,
            "closed": closed,
            "closed_since": closed_since,
            "opened_late": opened,
            "opened_since": opened_since
        })

    # Create a dataframe from the results
    summary = pd.DataFrame(results)

    # Merge the results with the station infromation based on the station ID
    stations = stations.merge(summary, on="SID", how="left")

    # Save the information as a csv file
    stations.to_csv(OUT / "stations_with_tags_test.csv", index=False)
    
if __name__ == "__main__":
    main()