import pandas as pd

def convert_numeric(df, date_column):
    cols=[i for i in df.columns if i not in [date_column]]
    for col in cols:
        df[col]=pd.to_numeric(df[col])
    df[date_column] = pd.to_datetime(df[date_column])
    return df

### mobility https://www.google.com/covid19/mobility/
df_mobility = pd.read_csv('data/test.csv')
df_mobility = df_mobility[df_mobility['country_region_code']=='BD']
df_mobility = df_mobility[['date','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline']]
df_mobility_mean = df_mobility[['retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline']]
df_mobility['mean'] = df_mobility_mean.mean(axis=1)
# print(df_mobility.head())
df_mobility = convert_numeric(df_mobility, 'date')
# df_mobility.plot(x='date')

df_mobility.to_csv('data/mobility.csv', index=False)

