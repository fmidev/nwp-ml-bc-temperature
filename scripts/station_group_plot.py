import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# Paths
STATION_FILE = Path.home() / "thesis_project" / "data" / "stations_with_tags.csv"
OBS_FILE = Path.home() / "thesis_project" / "data" / "observations.csv"
OUT = Path.home() / "thesis_project" / "figures" / "station_groups"
OUT.mkdir(parents=True, exist_ok=True)


def plot_station_group_proportions(stations, tag_col = "tag", closed_col = "closed", opened_col = "opened_late"):
    """
    A function to plot the proportions of stations in each group
        Params: 
            stations = Dataframe of the stations
            tag_col = Name of the statin tag column
            closed_col = Name of the column for the closed stations
            opened_col = Name of the column for the stations that opened late
        Saves the plot of the group proportions"""    

    # Create a dataframe copy of the stations
    df = stations.copy()

    # Normalize tag spelling/arrow variants and fill missing
    def _norm_tag(x):
        if pd.isna(x):
            return "mixed"
        s = str(x).strip().lower().replace("->", "→").replace(" ", "")
        if s.startswith("3h→1"):
            return "3h→1h"
        if s in {"1h", "3h", "mixed", "3h→1h"}:
            return s
        return "mixed"

    # Apply the tag normalization to the dataframe 
    df[tag_col] = df[tag_col].apply(_norm_tag)

    # Ensure boolean columns exist
    if closed_col not in df.columns:
        df[closed_col] = False
    if opened_col not in df.columns:
        df[opened_col] = False

    # Build final category with precedence: closed > opened late> tag
    cat = np.where(df[closed_col].fillna(False), "closed",
          np.where(df[opened_col].fillna(False), "opened_late", df[tag_col]))
    df["category"] = cat

    #  Order + counts -> proportions
    order = ["1h", "3h", "3h→1h", "mixed", "opened_late", "closed"]  #
    counts = df["category"].value_counts().reindex(order, fill_value=0)
    total = counts.sum() if counts.sum() > 0 else 1
    props = counts / total

    # Print a small summary table
    summary = pd.DataFrame({"count": counts, "proportion": (props * 100).round(1).astype(str) + "%"})
    print(summary)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(props.index, props.values)
    for b, p in zip(bars, props.values):
        ax.text(b.get_x() + b.get_width()/2, p + 0.01, f"{p*100:.1f}%", ha="center", va="bottom")

    # Plot parameters
    ax.set_ylim(0, max(0.05, props.values.max() + 0.08))
    ax.set_ylabel("Proportion of stations")
    ax.set_title("Station groups (share of total)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)

    # Save the plot
    plt.tight_layout()
    plt.savefig(OUT / "station_group.png", dpi=150, bbox_inches="tight")




def _norm_tag(x):
    """
    A helper function to normalize tag spelling/arrow variants and fill missing
        Param:
            x = tag to be normalized
        Returns: Normalized tag"""
    if pd.isna(x): 
        return "mixed"
    s = str(x).strip().lower().replace("->","→").replace(" ", "")
    if s.startswith("3h→1"):
        return "3h→1h"
    if s in {"1h","3h","mixed","3h→1h"}:
        return s
    return "mixed"


def monthly_nan_heatmap_for_tag(stations, obs, tag_filter = "3h→1h", value_col = "obs_TA", station_col = "SID", time_col = "obstime", months_back: int | None = None, include_closed = True, save_name: str | None = None):
    """
    Heatmap of monthly NaN counts for stations with a specific tag (default: '3h→1h').
    One cell = number of NaNs for (station, month).
        Params: 
            stations = Dataframe of station information
            obs = Dataframe of the observations
            tag_filter = Choose the tag; ('1h', '3h', '3h→1h', 'mixed') 
            value_col = Name of the observation column
            station_col = Name of the station ID column
            time_col = Name of the observation time column 
            months_back = How many months to show; None = show all months in data
            include_closed = Include stations that have been closed
            save_name = The name for which to save the plot
        Saves the created plot
    """

    # Normalize tags and filter stations
    st = stations.copy()
    st["tag_norm"] = st["tag"].apply(_norm_tag) if "tag" in st.columns else "mixed"
    if not include_closed and "closed" in st.columns:
        st = st[~st["closed"].fillna(False)]
    wanted_sids = set(st.loc[st["tag_norm"] == tag_filter, station_col])

    if not wanted_sids:
        print(f"No stations with tag '{tag_filter}'.")
        return

    # Prepare the observations, restrict to wanted stations
    df = obs.loc[obs[station_col].isin(wanted_sids), [station_col, time_col, value_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values([station_col, time_col])

    # Print if there is no observations available for the selected stations
    if df.empty:
        print("No observation rows available for the selected stations.")
        return

    # Month as Period
    df["month"] = df[time_col].dt.to_period("M")

    # Monthly NaN COUNT per station
    grp = df.groupby([station_col, "month"])
    monthly = grp[value_col].agg(
        total="size",
        n_nan=lambda x: x.isna().sum()
    ).reset_index()

    if monthly.empty:
        print("No monthly aggregates available.")
        return

    # Full month range & optional trimming 
    all_months = pd.period_range(monthly["month"].min(), monthly["month"].max(), freq="M")
    if months_back is not None and months_back > 0:
        all_months = all_months[-months_back:]  # last N months

    # Pivot to stations x months with NaN COUNTS
    pivot = monthly.pivot(index=station_col, columns="month", values="n_nan")
    pivot = pivot.reindex(columns=all_months, fill_value=np.nan)

    # Sort stations by overall NaN count (desc) for readability
    station_order = pivot.sum(axis=1, skipna=True).sort_values(ascending=False).index
    pivot = pivot.loc[station_order]

    # Labels
    col_labels = [str(p) for p in pivot.columns]

    # Plot heatmap (counts, not rates) 
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*0.6), max(5, len(pivot)*0.28)))
    im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")  # default colormap

    # Plot parameters
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.astype(str))
    ax.set_xlabel("Month")
    ax.set_ylabel("Station (SID)")
    ax.set_title(f"Monthly NaN COUNT for stations tagged '{tag_filter}'")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("NaN count (per month)")

    # Save the plot
    plt.tight_layout()
    plt.savefig(OUT / save_name, dpi=150, bbox_inches="tight")


def main():

    # Load station information into a dataframe
    stations = pd.read_csv(STATION_FILE) 

    # Load the observations into a dataframe
    obs = pd.read_csv(OBS_FILE)

    # Name of the plot to be saved
    pic_name = "mixed.png"

    # Make the plots
    plot_station_group_proportions(stations, tag_col="tag", closed_col="closed", opened_col="opened_late")
    monthly_nan_heatmap_for_tag(stations, obs, tag_filter = "mixed", value_col="obs_TA", station_col="SID", time_col="obstime", months_back=None, include_closed=True, save_name = pic_name)
        
if __name__ == "__main__":
    main()
