import os
import re
import glob
import pandas as pd
from pathlib import Path

# Paths
CSV_FOLDER = "/data/MOS_data/MOS_coef_2024_winter"
OUT = Path.home() / "thesis_project" / "data" / "mos_data"
OUT.mkdir(parents=True, exist_ok=True)

# Filename parser: station_<SID>_<00|12>_... ----
SID_AH_RE = re.compile(r"station_(?P<SID>\d+)_(?P<ah>00|12)_", re.IGNORECASE)

def parse_sid_ah(fname: str):
    "Parse the station ID and analysishour from the filename"
    m = SID_AH_RE.search(Path(fname).name)
    if not m:
        return None, None
    return m.group("SID"), int(m.group("ah"))

rows = []
  
all_csvs = sorted(glob.glob(os.path.join(CSV_FOLDER, "*_*_*_*_TA_*.csv")))
if not all_csvs:
    raise FileNotFoundError(f"No CSVs in {CSV_FOLDER}")


wide_rows = []   # collect per-file "leadtime × vars" wide frames

for csv_file in all_csvs:
    sid, ah = parse_sid_ah(csv_file)
    print(csv_file)
    if sid is None:
        print(f"[skip name mismatch] {csv_file}")
        continue

    df = pd.read_csv(csv_file, low_memory=False)

    # Pick the variable-name column in the ORIGINAL file
    if "Unnamed: 0" in df.columns:        
        var_col = "Unnamed: 0"
        first = df.columns[0]
        if first != var_col:
            col0 = df.iloc[:, 0]
            if pd.api.types.is_integer_dtype(col0) or col0.astype(str).str.fullmatch(r"\d+").all():
                df = df.drop(columns=[first])
    else:
        var_col = df.columns[0]           

    if var_col != "var":
        df = df.rename(columns={var_col: "var"})
    df["var"] = df["var"].astype(str).str.strip()

    # Leadtime columns (accept "00" or 0, 144, …)
    lt_cols = []
    for c in df.columns:
        if c == "var":
            continue
        name = str(c).strip()
        try:
            int(name)
            lt_cols.append(c)
        except ValueError:
            pass

    if not lt_cols:
        print(f"[skip no leadtimes] {csv_file}")
        continue

    # Produce WIDE table with one row per leadtime 
    # df: rows = var, columns = leadtimes; make rows = leadtime, cols = var
    wide_lead = (
        df.set_index("var")[lt_cols]     # columns = leadtime, index = var
          .T                             # rows = leadtime, cols = var
          .rename_axis("leadtime")       # index name
          .reset_index()                 # make leadtime a col
    )

    # Clean types
    wide_lead["leadtime"] = pd.to_numeric(
        wide_lead["leadtime"].astype(str).str.strip(), errors="coerce"
    ).astype("Int64")

    # Add keys
    wide_lead["SID"] = sid
    wide_lead["ah"] = ah

    # Arrange columns (keys first)
    key_cols = ["SID", "ah", "leadtime"]
    pred_cols = [c for c in wide_lead.columns if c not in key_cols]
    wide_lead = wide_lead[key_cols + pred_cols]

    # Drop rows with invalid leadtime
    wide_lead = wide_lead.dropna(subset=["leadtime"])

    wide_rows.append(wide_lead)

if not wide_rows:
    raise RuntimeError("No usable coefficient rows parsed.")

# Combine all files; average if duplicates (same SID, ah, leadtime) exist
wide_all = pd.concat(wide_rows, ignore_index=True)

# Optional: deterministic predictor order
pred_cols_sorted = sorted([c for c in wide_all.columns if c not in key_cols])
wide = wide_all[key_cols + pred_cols_sorted]

# Write WIDE outputs
csv_path     = OUT / "mos_2024_winter.csv"
wide.to_csv(csv_path, index=False)

print("Saved WIDE tables:")
print(" -", csv_path)
print("Shape (rows, cols):", wide.shape)
print(wide.head())

