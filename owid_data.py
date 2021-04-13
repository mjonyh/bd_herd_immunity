import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import numpy as np
from scipy.stats import linregress

def data_plot(x, y, color, label, ax=plt):
        mask = ~np.isnan(x) & ~np.isnan(y)
        stats = linregress(x[mask], y[mask])
        print(stats)

        m = stats.slope
        b = stats.intercept
        r2 = stats.rvalue*stats.rvalue
        p = stats.pvalue
        label = 'm: {:.2}, $R^2$: {:.2}'.format(m, r2)
        # label = '$R^2$: {:.2}'.format(r2)

        ax.scatter(x, y, marker='.', color=color, label=None)
        l, = ax.plot(x, m * x + b, color=color, label=label)
        # ax.fill_between(x, m*x+b+err, m*x+b-err, alpha=0.2, color=color)
        ax.set_title(label)
        return m

df = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
# df = pd.read_csv('owid-covid-data.csv')
df = df[df['location']=='Bangladesh']
df['date'] = pd.to_datetime(df['date'])

df.to_csv('data/tpr.csv', index=False)

fig, ax1 = plt.subplots(1)

color = 'tab:blue'
ax1.plot(df['date'], df['new_cases']/1000, color=color)
l1, = ax1.plot(df['date'], df['new_cases']/1000, color=color, marker='.', label='Daily Cases')

ax1.plot(df['date'], df['new_tests']/1000, color=color)
l2, = ax1.plot(df['date'], df['new_tests']/1000, color=color, marker='^', label='Daily Tests')
ax1.set_ylim(0, 27)
ax1.set_ylabel("Count (in 1,000)",color=color)
ax1.tick_params(axis='y',labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'

ax2.plot(df['date'], df['positive_rate']*100, color=color)
l3, = ax2.plot(df['date'], df['positive_rate']*100, color=color, marker='.', label='Positive Test Rate (in %)')
ax2.set_ylim(0, 27)
ax2.set_ylabel("Positive Test Rate (in %)",color=color)
ax2.tick_params(axis='y',labelcolor=color)

ax2.tick_params(axis='x', rotation=45)

ax2.legend(handles=[l1, l2, l3], loc=1, framealpha=0.5)
fig.tight_layout()

# df = df[df['new_cases'] > 1000]
df.positive_rate = df.positive_rate 
fig3, axs3 = plt.subplots(1, 3, figsize=(24, 5))
data_plot(df.new_tests, df.new_cases, 'r', None, axs3[0])
data_plot(df.new_cases, df.positive_rate, 'r', None, axs3[1])
data_plot(df.new_cases, df.positive_rate*df.new_cases, 'r', None, axs3[2])
axs3[0].set_ylabel('Daily Cases')
axs3[0].set_xlabel('Daily Tests')
axs3[1].set_xlabel('Daily Cases')
axs3[1].set_ylabel('TPR')
axs3[2].set_xlabel('Daily Cases')
axs3[2].set_ylabel('TPR x Cases')

# plt.figure()

# df.plot(kind='scatter', x='new_cases', y='positive_rate')


# df_rt_1 = pd.read_csv('data/rt_1.csv')
# df_rt_1['Date'] = pd.to_datetime(df_rt_1['Date'])
# df_rt_1['smooth_ml'] = df_rt_1['ML'].rolling(7).mean()

# fig2, axs = plt.subplots(1)
# df.plot(x='date', y='reproduction_rate', ax=axs)
# df_rt_1.plot(x='Date', y='ML', ax=axs)
# df_rt_1.plot(x='Date', y='smooth_ml', ax=axs)
plt.show()


# import plotly.graph_objects as go

# fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=df['date'],
#     y=df['positive_rate']*100,
#     name="Positive Test Rate (in %)"
# ))


# fig.add_trace(go.Scatter(
#     x=df['date'],
#     y=df['new_cases'],
#     name="Daily Confirmed Cases",
#     yaxis="y2"
# ))

# fig.add_trace(go.Scatter(
#     x=df['date'],
#     y=df['new_tests'],
#     name="Daily Tests",
#     yaxis="y3"
# ))


# # Create axis objects
# fig.update_layout(
#     xaxis=dict(
#         domain=[0, 0.75]
#     ),
#     yaxis=dict(
#         title="Positive Test Rate (in %)",
#         titlefont=dict(
#             color="#1f77b4"
#         ),
#         tickfont=dict(
#             color="#1f77b4"
#         ),
#         range=[0,27],

#     ),
#     yaxis2=dict(
#         title="Daily Confirmed Cases",
#         titlefont=dict(
#             color="#d62728"
#         ),
#         tickfont=dict(
#             color="#d62728"
#         ),
#         anchor="free",
#         overlaying="y",
#         side="right",
#         position=0.85,
#         range=[0,5400],
#     ),
#     yaxis3=dict(
#         title="Daily Tests",
#         titlefont=dict(
#             color="green"
#         ),
#         tickfont=dict(
#             color="green"
#         ),
#         anchor="x",
#         overlaying="y",
#         side="right",
#         range=[0,27000],
#     )
# )

# # Update layout properties
# fig.update_layout(
#     title_text="Positive Test Rate",
#     # width=800,
# )


# fig.show()

# import chart_studio
# import chart_studio.plotly as py

# username='mjonyh'
# api_key='VVmftIeOEKQFkLHcldpQ'

# chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
# py.plot(fig, filename='Positive Test Rate')
