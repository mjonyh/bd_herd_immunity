#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df_cases_real = pd.read_csv('data/districts_real.csv')
df_cases_sim = pd.read_csv('data/districts_sim.csv')

df_rt_real = pd.read_csv('data/districts_real_rt_gr_dt.csv')
df_rt_sim = pd.read_csv('data/districts_sim_rt_gr_dt.csv')

df_cases_real.date = pd.to_datetime(df_cases_real.date)
df_cases_sim.days_sim = pd.to_datetime(df_cases_sim.days_sim)
df_rt_real.date = pd.to_datetime(df_rt_real.date)
df_rt_sim.date = pd.to_datetime(df_rt_sim.date)


### select district_name from Bangladesh map by clicking on the district
district_name = 'Dhaka'

df_district_real = df_cases_real[df_cases_real.district == district_name]
df_district_sim = df_cases_sim[df_cases_sim.district == district_name]
df_district_rt_real = df_rt_real[df_rt_real.district == district_name]
df_district_rt_sim = df_rt_sim[df_rt_sim.district == district_name]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative Cases', color=color)
ax1.semilogy(df_district_real.date, df_district_real.confirmed, color=color, marker='.')
ax1.semilogy(df_district_sim.days_sim, df_district_sim.confirmed_sim, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(bottom=0.1, top=1e6)

ax2 = ax1.twinx()  

color = 'tab:blue'
ax2.set_ylabel('$R_t$', color=color)  
ax2.plot(df_district_rt_real.date, df_district_rt_real.ML, color=color, marker='.')
ax2.plot(df_district_rt_sim.date, df_district_rt_sim.ML, color=color)
ax2.fill_between(df_district_rt_sim.date, df_district_rt_sim.High_90, df_district_rt_sim.Low_90, color=color, alpha=0.3)


ax2.tick_params(axis='y', labelcolor=color)

ax2.set_ylim(bottom=0, top=7)

fig.tight_layout()  

plt.show()
