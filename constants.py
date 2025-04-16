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

DATE_RANGE = pd.date_range(start="1995-01-01", end="2025-01-01", freq='D')

DROP_COLUMNS = [
    "STATION",      # NOAA Station ID
    "ACMH",         # Average cloudiness midnight to midnight from manual observations (percent) 
    "DAPR",         # Number of days included in the multiday precipitation total (MDPR) 
    "DASF",         # Number of days included in the multiday snowfall total (MDSF)
    "MDPR",         # Multiday precipitation total (mm or inches as per user preference; use with DAPR and DWPR, if available)
    "MDSF",         # Multiday snowfall total (mm or inches as per user preference)
    "PSUN",         # Daily percent of possible sunshine (percent)
    "TOBS",         # Temperature at the time of observation (Fahrenheit or Celsius as per user preference)
    "WV01",         # Fog, ice fog, or freezing fog (may include heavy fog) in the vicinity
    "WV03",         # Thunder in the vicinity
    "WV18",         # Snow or ice crystals in the vicinity
    "WV20"          # Rain or snow shower in the vicinity
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
