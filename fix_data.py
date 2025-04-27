import pandas as pd, numpy as np, os, math
from pathlib import Path

RAW_PATH = "data/merged_pca_11_cmp_7d.csv"
OUT_PATH = "data/weekly_dataset1.csv"
BAD_STATIONS = ["AUBURN", "ESCANABA"]
BASELINE_YEARS = (1995, 2014)

# Adjusted thresholds
P_HIGH, P_LOW  = 0.90, 0.10  # relaxed to 90%/10% for better frequency

# Updated Definitions
PCTL_RULES = {
    "TMAX_HOT": {"base":"TMAX", "tail":"high", "p":P_HIGH},
    "TMAX_COLD":{"base":"TMAX", "tail":"low",  "p":P_LOW},
    "TMIN_TROP":{"base":"TMIN", "tail":"high", "p":P_HIGH},
    "TMIN_FROST":{"base":"TMIN","tail":"low",  "p":P_LOW},
    "PRCP_P95": {"base":"PRCP", "tail":"high", "p":P_HIGH},
    "SNOW_P95": {"base":"SNOW", "tail":"high", "p":P_HIGH},
    "SNWD_P90": {"base":"SNWD", "tail":"high", "p":0.90},
    "AWND_P95": {"base":"AWND", "tail":"high", "p":P_HIGH},
}
ABS_RULES = {
    "TMIN_LT0": {"base":"TMIN", "op":"<=", "value":0.0},
}

# Kept Compound Definitions
PREDEF = {
    "HEAT_DRY":["EXT_TMAX_HOT"],
    "THAW_FREEZE":["TMAX_HOT_T","TMIN_FROST_TPLUS1"],
    "RAIN_ON_SNOW":["EXT_PRCP_P95","EXT_SNWD_P90_LAG1","TMAX_ABOVE0"],
    "BACK2BACK_RAIN":["EXT_PRCP_P95"],
}

# -------------------- SCRIPT --------------------
print("1) Load daily …")
df = pd.read_csv(RAW_PATH, parse_dates=["DATE"])
df = df[df.DATE >= "1995-01-02"]
df = df[~df.STATION.isin(BAD_STATIONS)].set_index("DATE")

# Remove EXT_/COMPOUND_
df = df.loc[:, ~df.columns.str.startswith(("EXT_", "COMPOUND_"))]

# Create EXT flags
print("2) Create EXT_* flags …")
baseline = df[df.index.year.isin(range(*BASELINE_YEARS))]
thresholds = {tag: baseline.groupby("STATION")[r["base"]].quantile(r["p"]) 
              for tag,r in PCTL_RULES.items()}
for tag,r in PCTL_RULES.items():
    thr = thresholds[tag]
    cond = df[r["base"]] >= df.STATION.map(thr) if r["tail"]=="high" else df[r["base"]] <= df.STATION.map(thr)
    df[f"EXT_{tag}"] = cond.astype("int8")
for tag,r in ABS_RULES.items():
    cond = df[r["base"]] <= r["value"] if r["op"]=="<=" else df[r["base"]] >= r["value"]
    df[f"EXT_{tag}"] = cond.astype("int8")

# Lagged features
print("3) Create lagged EXT features …")
df["TMAX_HOT_T"] = df.groupby("STATION")["EXT_TMAX_HOT"].shift(0).fillna(0).astype("int8")
df["TMIN_FROST_TPLUS1"] = df.groupby("STATION")["EXT_TMIN_FROST"].shift(-1).fillna(0).astype("int8")
df["EXT_SNWD_P90_LAG1"] = df.groupby("STATION")["EXT_SNWD_P90"].shift(7).fillna(0).astype("int8")
df["TMAX_ABOVE0"] = (df["TMAX"] > 0).astype("int8")

# 7-day rolling EXT_
print("4) Expand EXT_* with rolling …")
ext_cols = [c for c in df if c.startswith("EXT_") or c in ["TMAX_HOT_T","TMIN_FROST_TPLUS1","EXT_SNWD_P90_LAG1","TMAX_ABOVE0"]]
df[ext_cols] = df.groupby("STATION")[ext_cols].transform(lambda s: s.rolling(7, center=True, min_periods=1).max())

# Weekly aggregation
print("5) Aggregate weekly …")
weekly = df.groupby("STATION").resample("W-MON", label="left").agg({
    **{col:"mean" for col in ["ACSH","AWND","SNWD","TAVG","TSUN","WDF5","WDFG"]},
    **{col:"sum" for col in ["PRCP","SNOW"]+ext_cols},
    "TMAX":"max", "TMIN":"min",
    "LATITUDE":"first", "LONGITUDE":"first", "ELEVATION":"first"
}).reset_index().rename(columns={"DATE":"WEEK_START"})

# Compounds & shift next week
print("6) Tag & Shift COMPOUND events …")
for tag,conds in PREDEF.items():
    weekly[f"COMPOUND_{tag}"] = weekly[conds].all(1).astype("int8")
for c in weekly.filter(like="COMPOUND_"):
    weekly[f"{c}_next"] = weekly.groupby("STATION")[c].shift(-1)
weekly.dropna(inplace=True)

# Split
val_sts = np.random.RandomState(1).choice(weekly.STATION.unique(), round(weekly.STATION.nunique()*0.2), replace=False)
weekly["SPLIT"] = np.where(weekly.STATION.isin(val_sts),"val","train")

weekly.to_csv(OUT_PATH,index=False)
print(f"Saved {OUT_PATH}")
