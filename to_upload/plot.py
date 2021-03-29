import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

start_date = pd.to_datetime('6/1/2020')
end_date = pd.to_datetime('7/1/2020')

df_sim = pd.read_csv('data/Bangladesh_90_2.csv')
df_sim['date'] = pd.to_datetime(df_sim['days_sim'])
df_real = pd.read_csv('data/Bangladesh_30_1.csv')
df_real['date'] = pd.to_datetime(df_real['days'])
df_arima = pd.read_csv('data/arima.csv')
df_arima.columns = ['day', 'confirmed_arima']
df_arima['date'] = pd.to_datetime(df_arima['day'])
df_sarimax = pd.read_csv('data/sarimax.csv')
df_sarimax.columns = ['day', 'confirmed_sarimax']
df_sarimax['date'] = pd.to_datetime(df_sarimax['day'])
# print(df_arima[df_arima['date'] < start_date])

df_sim = df_sim[(df_sim['date'] >= start_date) & (df_sim['date'] < end_date)]
df_real = df_real[(df_real['date'] >= start_date) & (df_real['date'] < end_date)]
df_arima = df_arima[(df_arima['date'] >= start_date) & (df_arima['date'] < end_date)]
df_sarimax = df_sarimax[(df_sarimax['date'] >= start_date) & (df_sarimax['date'] < end_date)]
# print(df_sim, df_arima)

df_merged = pd.merge(pd.merge(df_sim, df_real, on='date'), df_arima, on='date')
df_merged = pd.merge(df_merged, df_sarimax, on='date')
df_merged['error_confirmed_ukf'] = np.abs(df_merged['confirmed'] - df_merged['confirmed_sim'])*100/df_merged['confirmed']
df_merged['error_confirmed_arima'] = np.abs(df_merged['confirmed'] - df_merged['confirmed_arima'])*100/df_merged['confirmed']
df_merged['error_confirmed_sarimax'] = np.abs(df_merged['confirmed'] - df_merged['confirmed_sarimax'])*100/df_merged['confirmed']
# df_merged['confirmed', 'confirmed_sim', 'confirmed_arima'] = df_merged['confirmed', 'confirmed_sim', 'confirmed_arima'] / 1000

elements = ['confirmed', 'confirmed_sim', 'confirmed_arima', 'confirmed_sarimax']

for element in elements:
    df_merged[element] = df_merged[element]/1000

fig, axs = plt.subplots(2,1, sharex=True)
df_merged.plot(x='date', y=['confirmed', 'confirmed_sim', 'confirmed_arima'],
        style=['.', '-', '-'],
        color=['r', 'b', 'g'],
        label=['Real', 'UKF', 'ARIMA'],
        ax=axs[0])
df_merged.plot(x='date', y=['error_confirmed_ukf', 'error_confirmed_arima'], marker='.',
        color=['b', 'g'],
        label=['UKF', 'ARIMA'],
        ax=axs[1])

axs[0].set_ylabel('Confirmed Cases (x1K)')
axs[1].set_ylabel('Error (%)')
axs[1].set_yticks([0, 5, 10, 15, 20])

# print(df_merged)
# date_array  = df_sim['date'].to_numpy()
# confirmed_array  = df_real['confirmed'].to_numpy()
# sim_array  = df_sim['confirmed_sim'].to_numpy()


# ax = df_real.plot(x='date', y='confirmed')
# df_sim.plot(x='date', y='confirmed_sim', ax=ax)
# plt.figure()
# plt.plot(date_array, np.abs(confirmed_array - sim_array)*100/confirmed_array, 'o')

plt.show()

