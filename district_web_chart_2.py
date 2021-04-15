#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df_rt_real = pd.read_csv('data/districts_real_rt_gr_dt.csv')
df_rt_sim = pd.read_csv('data/districts_sim_rt_gr_dt.csv')

df_rt_real.date = pd.to_datetime(df_rt_real.date)
df_rt_sim.date = pd.to_datetime(df_rt_sim.date)


### select district_name from Bangladesh map by clicking on the district
district_name = 'Dhaka'

df_district_rt_real = df_rt_real[df_rt_real.district == district_name]
df_district_rt_sim = df_rt_sim[df_rt_sim.district == district_name]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Growth Rate', color=color)
ax1.semilogy(df_district_rt_real.date, df_district_rt_real.growth_rate_ML, color=color, marker='.')
ax1.semilogy(df_district_rt_sim.date, df_district_rt_sim.growth_rate_ML, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(bottom=0.001)

ax2 = ax1.twinx()  

color = 'tab:blue'
ax2.set_ylabel('Doubling Time', color=color)  
ax2.plot(df_district_rt_real.date, df_district_rt_real.doubling_time_ML, color=color, marker='.')
ax2.plot(df_district_rt_sim.date, df_district_rt_sim.doubling_time_ML, color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax2.set_ylim(bottom=0, top=400)

fig.tight_layout()  

plt.show()
