import pandas as pd

df_2020 = pd.read_csv('2020_BD_Region_Mobility_Report.csv')
df_2021 = pd.read_csv('2021_BD_Region_Mobility_Report.csv')

df = df_2020.append(df_2021, ignore_index = True)

# print(df['date'])

df.to_csv('data/google_mobility.csv', index=False)
