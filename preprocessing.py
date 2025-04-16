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

DATE_RANGE = pd.date_range(start="1995-01-01", end="2025-01-01", freq='D')


def process_data(name):
    df = pd.read_csv(f"data/raw/{name}.csv")
    
    # Drop unnecessary columns
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    # Convert date column
    df["DATE"] = pd.to_datetime(df["DATE"])

    # Separate stations by name
    groups = df.groupby("NAME")

    # Export each individual city's weather data
    for station_name, station_data in groups:
        # Shorten station name
        for city in CITY_NAMES:
            if city in station_name:
                station_name = city
                station_data["NAME"] = city
                CITY_NAMES.remove(city)
                break
        
        # Fill in missing dates
        station_data.set_index("DATE", inplace=True)
        station_data.sort_index(inplace=True)
        station_data = station_data.reindex(DATE_RANGE)

        # For missing dates that got added back, get the name and geographic location back
        station_data["NAME"] = station_data["NAME"].fillna(station_name)
        station_data["LATITUDE"] = station_data["LATITUDE"].fillna(station_data["LATITUDE"].mode()[0])
        station_data["LONGITUDE"] = station_data["LONGITUDE"].fillna(station_data["LONGITUDE"].mode()[0])
        station_data["ELEVATION"] = station_data["ELEVATION"].fillna(station_data["ELEVATION"].mode()[0])

        station_data.to_csv(f"data/processed/{station_name}.csv")


if __name__ == "__main__":
    process_data("CMCTB")
    process_data("RGEDM")
    process_data("SGASS")
    process_data("FSALH")
    process_data("MHWTM")
    process_data("CHIMM")
    process_data("SBGAE")
