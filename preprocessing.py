import pandas as pd


def load_data(name):
    return pd.read_csv(f"data/{name}.csv")


def separate_stations():
    pass


if __name__ == "__main__":
    df1 = load_data("CMCTB")

    print(df1.head())
