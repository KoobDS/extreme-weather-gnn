# tag_extremes_subset.py
# ------------------------------------------------------------
import pandas as pd
from typing import Dict, Tuple

# ------------------- 1.  GLOBAL SETTINGS --------------------
BASELINE_YEARS: Tuple[int, int] = (1995, 2014)   # change if needed
P_HIGH, P_LOW = 0.95, 0.05

# ---- percentile‑based rules (tail = "high" | "low") --------
PCTL_RULES: Dict[str, Dict] = {
    # cloudiness & sunshine
    "ACMH_CLOUDY": {"base": "ACMH", "tail": "high", "p": P_HIGH},
    "ACMH_CLEAR":  {"base": "ACMH", "tail": "low",  "p": P_LOW},
    "ACSH_CLOUDY": {"base": "ACSH", "tail": "high", "p": P_HIGH},
    "ACSH_CLEAR":  {"base": "ACSH", "tail": "low",  "p": P_LOW},
    "PSUN_DARK":   {"base": "PSUN", "tail": "low",  "p": P_LOW},
    "PSUN_BRIGHT": {"base": "PSUN", "tail": "high", "p": P_HIGH},
    "TSUN_DARK":   {"base": "TSUN", "tail": "low",  "p": P_LOW},
    "TSUN_BRIGHT": {"base": "TSUN", "tail": "high", "p": P_HIGH},

    # temperature
    "TMAX_HOT":    {"base": "TMAX", "tail": "high", "p": P_HIGH},
    "TMAX_COLD":   {"base": "TMAX", "tail": "low",  "p": P_LOW},
    "TMIN_TROP":   {"base": "TMIN", "tail": "high", "p": P_HIGH},
    "TMIN_FROST":  {"base": "TMIN", "tail": "low",  "p": P_LOW},
    "TAVG_HOT":    {"base": "TAVG", "tail": "high", "p": P_HIGH},
    "TAVG_COLD":   {"base": "TAVG", "tail": "low",  "p": P_LOW},

    # precipitation & snow
    "PRCP_P95":    {"base": "PRCP", "tail": "high", "p": P_HIGH},
    "SNOW_P95":    {"base": "SNOW", "tail": "high", "p": P_HIGH},
    "SNWD_P90":    {"base": "SNWD", "tail": "high", "p": 0.90},
    "WESD_P90":    {"base": "WESD", "tail": "high", "p": 0.90},

    # wind speeds
    "AWND_P95":    {"base": "AWND", "tail": "high", "p": P_HIGH},
    "WSF1_P95":    {"base": "WSF1", "tail": "high", "p": P_HIGH},
    "WSF2_P95":    {"base": "WSF2", "tail": "high", "p": P_HIGH},
    "WSF5_P95":    {"base": "WSF5", "tail": "high", "p": P_HIGH},
    "WSFG_P95":    {"base": "WSFG", "tail": "high", "p": P_HIGH},
}

# ---- absolute cut‑off rules --------------------------------
ABS_RULES: Dict[str, Dict] = {
    "PRCP_20MM":   {"base": "PRCP", "op": ">=", "value": 20.0},   # mm
    "SNOW_10CM":   {"base": "SNOW", "op": ">=", "value": 100.0}, # mm
    "AWND_GALE":   {"base": "AWND", "op": ">=", "value": 17.0},  # m/s
    "WSFG_DAMG":   {"base": "WSFG", "op": ">=", "value": 25.0},  # m/s
    "TMAX_GT35":   {"base": "TMAX", "op": ">=", "value": 35.0},  # °C
    "TMIN_LT0":    {"base": "TMIN", "op": "<=", "value": 0.0},   # °C
}

# ------------------- 2.  MAIN HELPER ------------------------
def add_extreme_flags(df: pd.DataFrame,
                      baseline_years: Tuple[int, int] = BASELINE_YEARS,
                      verbose: bool = False) -> pd.DataFrame:
    """
    Add EXT_* boolean columns for every rule whose 'base' column exists
    in *df*. Missing columns are ignored.
    Assumes *df* has DATE (datetime64) and STATION columns.
    """
    out = df.copy()
    present_cols = set(out.columns)

    # --- 2.1  build thresholds for percentile rules ----------
    base = out[out.DATE.dt.year.between(*baseline_years)]
    thresholds = {}
    for tag, rule in PCTL_RULES.items():
        col = rule["base"]
        if col not in present_cols:
            if verbose: print(f"[skip] {tag} ‑‑ missing {col}")
            continue
        thresholds[tag] = base.groupby("STATION")[col].quantile(rule["p"])

    # --- 2.2  apply percentile rules -------------------------
    for tag, rule in PCTL_RULES.items():
        if tag not in thresholds:    # missing base col
            continue
        tail = rule["tail"]
        col  = rule["base"]
        thr  = thresholds[tag]
        if tail == "high":
            out[f"EXT_{tag}"] = (out[col] >= out["STATION"].map(thr)).astype(int)
        else:
            out[f"EXT_{tag}"] = (out[col] <= out["STATION"].map(thr)).astype(int)

    # --- 2.3  apply absolute rules ---------------------------
    for tag, rule in ABS_RULES.items():
        col = rule["base"]
        if col not in present_cols:
            if verbose: print(f"[skip] {tag} ‑‑ missing {col}")
            continue
        op, val = rule["op"], rule["value"]
        if op == ">=":
            out[f"EXT_{tag}"] = (out[col] >= val).astype(int)
        else:
            out[f"EXT_{tag}"] = (out[col] <= val).astype(int)

    # --- 2.4  Weather‑type codes already 0/1 -----------------
    for col in out.columns:
        if col.startswith(("WT", "WV")) and out[col].dropna().isin([0,1]).all():
            out[f"EXT_{col}"] = out[col].fillna(0).astype(int)

    return out
