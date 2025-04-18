import pandas as pd

CITY_NAMES = [
    "ALPENA",
    "APPLETON",
    "AUBURN",
    "BAD AXE",
    "BUFFALO",
    "CHEBOYGAN",
    "CHICAGO",
    "CLEVELAND",
    "DETROIT",
    "ERIE",
    "ESCANABA",
    "FORT WAYNE",
    "GRAND MARAIS",
    "GRAND RAPIDS",
    "GREEN BAY",
    "HIBBING",
    "HOUGHTON LAKE",
    "HOUGHTON",
    "IRONWOOD",
    "LANSING",
    "MANISTEE",
    "MARINETTE",
    "MARQUETTE",
    "MILWAUKEE",
    "MONTELLO",
    "MUSKEGON",
    "ROCHESTER",
    "SAGINAW",
    "SAULT STE MARIE",
    "SOUTH BEND",
    "SUPERIOR",
    "SYRACUSE",
    "TOLEDO",
    "TRAVERSE CITY",
    "WATERTOWN"
]

COLUMN_ORDER = [
    "NAME",
    "LATITUDE",
    "LONGITUDE",
    "ELEVATION",
    "ACSH",
    "AWND",
    "PRCP",
    "SNOW",
    "SNWD",
    "TAVG",
    "TMAX",
    "TMIN",
    "TSUN",
    "WDF5",
    "WDFG",
    "WESD",
    "WSF5",
    "WSFG",
    "WT01",
    "WT02",
    "WT03",
    "WT04",
    "WT05",
    "WT06",
    "WT07",
    "WT08",
    "WT09",
    "WT10",
    "WT11",
    "WT13",
    "WT14",
    "WT15",
    "WT16",
    "WT17",
    "WT18",
    "WT19",
    "WT21",
    "WT22"
]

DATE_RANGE = pd.date_range(start="1995-01-01", end="2025-01-01", freq='D')

DROP_COLUMNS = [
    "STATION",      # NOAA Station ID
    "ACMH",         # Average cloudiness midnight to midnight from manual observations (percent) 
    "DAPR",         # Number of days included in the multiday precipitation total (MDPR) 
    "DASF",         # Number of days included in the multiday snowfall total (MDSF)
    "FMTM",
    "MDPR",         # Multiday precipitation total (mm or inches as per user preference; use with DAPR and DWPR, if available)
    "MDSF",         # Multiday snowfall total (mm or inches as per user preference)
    "PGTM",
    "PSUN",         # Daily percent of possible sunshine (percent)
    "TOBS",         # Temperature at the time of observation (Fahrenheit or Celsius as per user preference)
    "WDF1",
    "WDF2",
    "WSF1",
    "WSF2",
    "WV01",         # Fog, ice fog, or freezing fog (may include heavy fog) in the vicinity
    "WV03",         # Thunder in the vicinity
    "WV18",         # Snow or ice crystals in the vicinity
    "WV20"          # Rain or snow shower in the vicinity
]

NUMERICAL_COLUMNS = [
    "ACSH",
    "AWND",
    "PRCP",
    "SNOW",
    "SNWD",
    "TAVG",
    "TMAX",
    "TMIN",
    "TSUN",
    "WDF5",
    "WDFG",
    "WESD",
    "WSF5",
    "WSFG",
    "WT01",
    "WT02",
    "WT03",
    "WT04",
    "WT05",
    "WT06",
    "WT07",
    "WT08",
    "WT09",
    "WT10",
    "WT11",
    "WT13",
    "WT14",
    "WT15",
    "WT16",
    "WT17",
    "WT18",
    "WT19",
    "WT21",
    "WT22"
]

WEATHER_TYPES = [
    "FOG",
    "HEAVY_FOG",
    "THUNDER",
    "SLEET",
    "HAIL",
    "GLAZE",
    "DUST",
    "SMOKE",
    "BLOWING_SNOW",
    "TORNADO",
    "WIND",
    "MIST",
    "DRIZZLE",
    "FREEZING_DRIZZLE",
    "RAIN",
    "FREEZING_RAIN",
    "SNOW",
    "UNKNOWN",
    "GROUND_FOG",
    "ICE_FOG"
]
