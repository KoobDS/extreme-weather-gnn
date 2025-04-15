import pandas as pd


def load_data(name):
    df = pd.read_csv(f"data/{name}.csv")
    
    # Drop unnecessary columns
    df = df.drop(columns=["STATION", "WV01", "WV03"], errors="ignore")

    # Shorten name column (WIP)
    df["NAME"] = df["NAME"].str.split().str[0]

    return df


def separate_stations(df):
    station_groups = df.groupby("NAME")

    return station_groups


if __name__ == "__main__":
    df1 = load_data("CMCTB")

    print(df1)
    #separate_stations(df1)
