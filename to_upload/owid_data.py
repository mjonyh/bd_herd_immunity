import pandas as pd

df = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
# df = pd.read_csv('data/owid-covid-data.csv')
df = df[df['location']=='Bangladesh']
df['date'] = pd.to_datetime(df['date'])

df.to_csv('data/tpr.csv', index=False)

