
from pathlib import Path
import polars as pl
import numpy as np 
import pandas as pd



# Paths
HOME = Path.home()
DATA_GLOB = HOME / "thesis_project" / "data" / "ml_data_full"

# MOS CSVs (long format: SID, ah, leadtime, var, coef)
MOS  = HOME / "thesis_project" / "data" / "mos_data"


# Output
OUTDIR = HOME / "thesis_project" / "metrics"
OUTDIR.mkdir(parents=True, exist_ok=True)


# Columns / labels
SPLIT_COLUMN = "analysistime"  # or "validtime"
TEMP_FC   = "T2"
LABEL_OBS = "obs_TA"

# ID columns present in data
ID = ["SID", "analysistime", "validtime", "leadtime"]
MOS_VARS = ['D2', 'Intercept', 'LCC', 'MCC', 'MN2T', 'MSL', 'MX2T', 'SKT', 'T2', 'T2_ENSMEAN_MA1', 'T2_M1', 'T_925', 'T_925_M1', 'U10', 'V10']
TEMP_COLS = ['D2', 'MN2T', 'MX2T', 'SKT', 'T2', 'T2_ENSMEAN_MA1', 'T2_M1', 'T_925', 'T_925_M1']

# Output column for MOS
MOS_CORR_COL = "corrected_mos"

#------------------------
# Helper functions
# -----------------------

def safe_month(col):
    return pl.col(col).str.strptime(pl.Datetime, strict=False, exact=False).dt.month()



# ---------------------------
# MOS correction function
# ---------------------------

def add_mos_correction_absolute(df: pl.DataFrame, mos_wide: pl.DataFrame, mos_vars: list[str], eps: float = 1e-12) -> pl.DataFrame:
    """
    MOS predicts ABSOLUTE temperature:
        corrected_mos = coef__Intercept + Σ[active v] (coef__v * df[v])

    active v := coef__v is non-null and |coef__v| > eps, and df[v] is non-null
    If there are no active coefs for (SID, season, ah, leadtime), corrected_mos is null.
    """
    # Derive season/hour/lead (typed as in mos_wide)
    df = df.with_columns([
        pl.col("analysistime").str.strptime(pl.Datetime, strict=False, exact=False).dt.hour().cast(pl.Int32).alias("ah"),
        pl.col("leadtime").cast(pl.Int32),
        pl.col("SID").cast(pl.Utf8),
    ])

    # Left join MOS coefs
    dfj = df.join(mos_wide, on=["SID", "ah", "leadtime"], how="left")


    # Build term list over coefs that actually exist in the join
    terms = []
    active_flags = []
    used_vars_parts = []
    for v in mos_vars:
        coef_c = f"coef_{v}"
        if (coef_c in dfj.columns) and (v in dfj.columns):
            # active if coef is non-null and |coef| > eps AND the feature value exists
            is_active = (
                pl.col(coef_c).is_not_null() &
                (pl.col(coef_c).abs() > eps) &
                pl.col(v).is_not_null()
            )
            active_flags.append(is_active.alias(f"_act__{v}"))
            # contribute only when active; otherwise 0
            terms.append(
                (pl.col(coef_c).fill_null(0.0) *
                pl.col(v).cast(pl.Float64).fill_null(0.0))
            )
            used_vars_parts.append(
                pl.when(is_active).then(pl.lit(v)).otherwise(None)
            )

    # Number of active coefs actually used for this row
    if active_flags:
        n_used = (
            pl.sum_horizontal([af.cast(pl.Int8) for af in active_flags])
            .cast(pl.Int32)
            .alias("_n_coefs_nnz")
        )
    else:
        n_used = pl.lit(0, dtype=pl.Int32).alias("_n_coefs_nnz")

    used_vars = (
        pl.concat_list(used_vars_parts).list.drop_nulls().alias("_used_vars")
        if used_vars_parts else pl.lit([], dtype=pl.List(pl.Utf8)).alias("_used_vars")
    )

    # Intercept (if present, else 0 contribution)
    intercept = (
        pl.col("coef_Intercept").fill_null(0.0)
        if "coef_Intercept" in dfj.columns
        else pl.lit(0.0)
    ).alias("intercept")

    # Linear part
    linear_sum = (sum(terms) if terms else pl.lit(0.0)).alias("linear")
    
    """
    Debug part no longer needed put left to the script in case it is needed later
    # Attach temporarily for debug inspection
    df_debug = dfj.with_columns(linear_sum, intercept)


    # Collect rows where linear_sum < 200 K
    debug_rows = df_debug.filter(pl.col("linear") < 200).head(5)

  

    if debug_rows.height > 0:
        print("\n[DEBUG] Found rows with linear_sum < 200K (only active coefs shown):")
        for row in debug_rows.iter_rows(named=True):
            sid = row.get("SID", "?")
            lt  = row.get("leadtime", "?")
            linear_val = row["linear"]
            intercpt = row["intercept"]
            print(f"  Station {sid}, leadtime {lt}, corrected={linear_val:.3f}, intercept={intercpt}")

            # print only active coefficients — coef != 0, not null, and abs(coef) > eps
            for v in mos_vars:
                coef_c = f"coef_{v}"
                if coef_c in row and v in row:
                    coef_val = row[coef_c]
                    var_val = row[v]
                    if coef_val is not None and abs(coef_val) > eps and var_val is not None:
                        #print(f"    {v:15s} coef={coef_val:10.4f}  value={var_val:10.4f}")
                        print(f"    {v:15s} coef={coef_val:10.4f}  value={var_val:10.4f}  contrib={coef_val * var_val:10.4f}")
            print("-" * 60)"""

    # Predicted temperature
    pred_expr = (intercept + linear_sum).alias("corrected_mos")

    # Final: only keep prediction when we had any active coef; else null
    corrected = pl.when((n_used > 0) & (pred_expr >= 200)).then(pred_expr).otherwise(None).alias("corrected_mos")

    out = (
        dfj.with_columns([n_used, corrected, intercept, used_vars, linear_sum])
           .drop([c for c in dfj.columns if c.startswith("_act__")])  
    )

    return out




def main():

    # Load the MOS coefficients
    mos_wide = pl.read_csv(MOS / "mos_2024_winter.csv")

    mos_wide = mos_wide.with_columns(pl.col("SID").cast(pl.Utf8))

    # Rename coefficient columns
    mos_wide = mos_wide.rename(
        {name: f"coef_{name}" for name in mos_wide.columns if name not in ["SID", "ah", "leadtime"]}
    )

    # Moths in the chosen season for loading coeffs
    months = ["12", "01", "02"]

    accumulated = []

    # Loop through the months 
    for month in months:

        print(f"[MOS] Processing month {month}...")

        if month == "12":
            year = "2024" 
        else:
            year = "2025"

        file = DATA_GLOB / f"ml_data_full_{year}-{month}.parquet"

        print(file)

        lf0 = pl.scan_parquet(str(file))
        
        # select only columns that actually exist in this file
        schema = lf0.collect_schema()          # cheap schema introspection
        file_cols = set(schema.keys())

        # Get the needed columns
        need = set(ID + ["leadtime", LABEL_OBS] + MOS_VARS)
        select_cols = [c for c in need if c in file_cols]
        if not select_cols:
            continue

        lf = lf0.select(select_cols)
        lf = lf.with_columns(pl.col(SPLIT_COLUMN).cast(pl.Utf8))

        # Sort by unique SID, analysistime, leadtime
        lf = (
            lf.sort(["SID", "analysistime", "leadtime", "validtime"])
                .unique(subset=["SID", "analysistime", "leadtime"], keep="last")
        )


        # Create a dataframe
        df = lf.collect(engine="streaming")
        if df.height == 0:
            continue

        # Require base fc
        if TEMP_FC in df.columns:
            df = df.filter(pl.col(TEMP_FC).is_not_null())
        if df.height == 0:
            continue
        
        # Add MOS correction 
        df = add_mos_correction_absolute(df, mos_wide, MOS_VARS)

        # Extra columns for checks 
        extra_cols = [c for c in ["_used_vars", "coef__Intercept", "linear"] if c in df.columns]

        # Keep relevant columns
        keep_cols = [
            c for c in (ID + [TEMP_FC, LABEL_OBS, MOS_CORR_COL] + extra_cols)
            if c in df.columns]

        if (TEMP_FC not in keep_cols) or (LABEL_OBS not in keep_cols):
            continue

        # Append data
        accumulated.append(df.select(keep_cols))

        if not accumulated:
            print(f"[MOS] No rows collected for month {month}; skipping.")
            continue


    # Concatenate the data 
    all_df = pl.concat(accumulated, how="vertical_relaxed").with_columns(
        month=safe_month("validtime")
    )

    # Only keep rows where MOS produced a value
    usable = all_df.filter(pl.col(MOS_CORR_COL).is_not_null())

    # Get used varaibles for checking purposes
    usable = usable.with_columns(
        pl.col("_used_vars")
        .list.join("|")  # Join list elements into a single string
        .alias("_used_vars_str")
    ).drop("_used_vars")

    # Sort by analysistime, leadtime
    usable = usable.sort(["analysistime", "leadtime"])

    usable = usable.rename({"T2" : "raw_fc"})

    # Save the data
    tag = f"{SPLIT_COLUMN}_MOS_2025_winter"
    (OUTDIR / "mos").mkdir(parents=True, exist_ok=True)  # keep MOS outputs grouped

    usable.write_parquet(   OUTDIR / "mos" / f"eval_rows_{tag}.parquet")

    print("[MOS] Saved to:", OUTDIR / "mos")

if __name__ == "__main__":
    pl.Config.set_tbl_rows(20)
    main()
