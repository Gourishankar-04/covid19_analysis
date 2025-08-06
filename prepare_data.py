import pandas as pd
from pathlib import Path

RAW = Path("data/covid_19_clean_complete.csv")
OUT = Path("data/covid_cases_prepared.csv")

def main():
    df = pd.read_csv(RAW)
    df = df.rename(columns={
        'Country/Region': 'country',
        'Date': 'date',
        'WHO Region': 'who_region',
        'Confirmed': 'confirmed',
        'Deaths': 'deaths',
        'Recovered': 'recovered',
        'Active': 'active'
    })
    df['date'] = pd.to_datetime(df['date'])
    if 'active' not in df.columns:
        df['active'] = df['confirmed'] - df['deaths'] - df['recovered']
    group_cols = ['date', 'country', 'who_region']
    df = df.groupby(group_cols, dropna=False)[['confirmed','deaths','recovered','active']].sum().reset_index()
    df = df.sort_values(['country','date'])
    df.to_csv(OUT, index=False)
    print(f"Prepared dataset saved to {OUT}")

if __name__ == "__main__":
    main()
