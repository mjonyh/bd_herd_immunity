import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns
sns.set()

import matplotlib as mpl
mpl.rcParams['figure.subplot.left'] = 0.083
mpl.rcParams['figure.subplot.right'] = 0.973
mpl.rcParams['figure.subplot.bottom'] = 0.16
mpl.rcParams['figure.subplot.top'] = 0.86

from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def data_plot(x, y, color, label, ax=plt):
    mask = ~np.isnan(x) & ~np.isnan(y)
    stats = linregress(x[mask], y[mask])
    print(stats)

    m = stats.slope
    b = stats.intercept
    # err = stats.stderr/2
    r2 = stats.rvalue*stats.rvalue
    p = stats.pvalue
    # label = 'm: {:.2}, $R^2$: {:.2}'.format(m, r2)
    label = '$R^2$: {:.2}'.format(r2)

    ax.scatter(x, y, marker='.', color=color, label=None)
    l, = ax.plot(x, m * x + b, color=color, label=label)
    # ax.fill_between(x, m*x+b+err, m*x+b-err, alpha=0.2, color=color)
    return l

import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls

def convert_numeric(df, date_column):
    cols=[i for i in df.columns if i not in [date_column]]
    for col in cols:
        df[col]=pd.to_numeric(df[col])
    df[date_column] = pd.to_datetime(df[date_column])
    return df

def annot_max(x,y, ax=None, color='k'):
    xmax = x.iloc[-1]
    ymax = y.iloc[-1]
    # xmax = x[np.argmax(y)]
    # ymax = y.max()
    # text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    text= "{:.2f}".format(ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec=color, lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60", color=color)
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top", color=color)
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.46), **kw)

def graph_plot(x, y, tag, ax):
    # plt.figure(tag[0])

    if(tag[1]=='plot'):
        ax.plot(x, y, tag[2], label=tag[3])
    elif(tag[1]=='semilogy'):
        ax.semilogy(x, y, tag[2], label=tag[3])


    if(tag[3]!=None):
        ax.legend()

    plt.xticks(rotation=45)
    plt.ylabel(tag[4])


'''
files required
tag = [figure_no, plot/semilogy, color+linestyle, label, ylabel]
'''

files = ['Bangladesh_90']

df_rt_1 = pd.read_csv('data/rt_1.csv')
df_rt_1 = convert_numeric(df_rt_1, 'Date')

df_rt_2 = pd.read_csv('data/rt_2.csv')
df_rt_2 = convert_numeric(df_rt_2, 'Date')
# # print(df_rt)

# df_owid = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
df_owid = pd.read_csv('owid-covid-data.csv')
df_owid = df_owid[df_owid['location']=='Bangladesh']
df_owid['date'] = pd.to_datetime(df_owid['date'])

### mobility https://www.google.com/covid19/mobility/
df_mobility = pd.read_csv('data/google_mobility.csv')
df_mobility = df_mobility[df_mobility['country_region_code']=='BD']
df_mobility = df_mobility[['date','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline']]
df_mobility_mean = df_mobility[['retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline']]
df_mobility['mean'] = df_mobility_mean.mean(axis=1)
# print(df_mobility.head())
df_mobility = convert_numeric(df_mobility, 'date')
# df_mobility.plot(x='date')

# df_mobility.plot(x='date', y='mean')


fig2, ax2 = plt.subplots(1, 1, figsize=(6.4, 4), sharex=True)

xmax_limit = '2021-05-01'


### Real data
df_1 = pd.read_csv('data/Bangladesh_90_1.csv', encoding='UTF-8')
df_1 = convert_numeric(df_1, 'days')

### Simulated data
df_2 = pd.read_csv('data/Bangladesh_90_2.csv', encoding='UTF-8')
df_2 = convert_numeric(df_2, 'days_sim')

### reproduction numbers
# if (i<1):

color = 'tab:red'

### mobility
tag = [12, 'plot', 'k-', 'Drops in Mobility (x10)', None]
graph_plot(df_mobility['date'], -df_mobility['mean']/10, tag, ax2)
tag = [12, 'plot', 'k.', None, None]
graph_plot(df_mobility['date'], -df_mobility['mean']/10, tag, ax2)

### growth
tag = [12, 'plot', 'b-', 'Daily Cases (x1K)', None]
graph_plot(df_1['days'], df_1['confirmed'].diff()/1000, tag, ax2)
tag = [12, 'plot', 'b.', None, None]
graph_plot(df_1['days'], df_1['confirmed'].diff()/1000, tag, ax2)
tag = [12, 'plot', 'b--', None, None]
graph_plot(df_2['days_sim'], df_2['confirmed_sim'].diff()/1000, tag, ax2)

### Test Positive Rate
tag = [12, 'plot', 'g-', 'Test Positive Rate (x10 in %)', None]
graph_plot(df_owid['date'], df_owid['positive_rate']*10, tag, ax2)
tag = [12, 'plot', 'g.', None, None]
graph_plot(df_owid['date'], df_owid['positive_rate']*10, tag, ax2)

### lock down
text_position = 7.5
text_position_down = 4.5
line_max = 10
line_min = -2
# ax2.vlines(x='2020-03-26', ymin=line_min, ymax=line_max, color='black', alpha=0.5)
# ax2.text('2020-03-27', text_position, '$L_1$', size=10)
# ax2.vlines('2020-04-26', ymin=line_min, ymax=line_max, color='k', alpha=0.5)
# ax2.text('2020-04-27', text_position, '$L_2$', color='black', size=10)
# ax2.vlines('2020-05-10', color='k', alpha=0.5, ymin=line_min, ymax=line_max)
# ax2.text('2020-05-11', text_position, '$L_3$', color='black', size=10)
# ax2.vlines('2020-05-30', color='k', alpha=0.5, ymin=line_min, ymax=line_max)
# ax2.text('2020-05-31', text_position, '$L_4$', color='black', size=10)
# ### Eid ul Adha
# ax2.vlines('2020-07-31', color='k', alpha=0.5, ymin=line_min, ymax=line_max)
# ax2.text('2020-08-01', text_position, '$L_5$', color='black', size=10)
# ### durga puja
# ax2.vlines('2020-10-26', color='k', alpha=0.5, ymin=line_min, ymax=line_max)
# ax2.text('2020-10-27', text_position, '$L_6$', color='black', size=10)
# ### No more free test
# ax2.vlines('2020-06-30', color='r', alpha=0.5, ymin=line_min, ymax=line_max)
# ax2.text('2020-07-01', text_position, '$G_1$', color='r', size=10)
# ### Winter
# ax2.vlines('2020-11-15', color='r', alpha=0.5, ymin=line_min, ymax=line_max)
# ax2.text('2020-11-16', text_position, 'Winter', color='r', size=10)

### Rt
tag = [12, 'plot', 'r-', '$R_t$', None]
graph_plot(df_rt_2['Date'], df_rt_2['ML'], tag, ax2)       
tag = [12, 'plot', 'r.', None, None]
graph_plot(df_rt_1['Date'], df_rt_1['ML'], tag, ax2)       
annot_max(df_rt_1['Date'], df_rt_1['ML'], ax2, 'r')

test = ax2.fill_between(df_rt_1['Date'], df_rt_1['Low_90'], df_rt_1['High_90'], alpha=0.5, color='r')

df = pd.read_csv('data/owid-covid-data.csv')
df = df[df['location']=='Bangladesh']
df['date'] = pd.to_datetime(df['date'])

ax2.set_xlim(xmin='2020-03-15', xmax=xmax_limit)
# ax2.legend(ncol=2, framealpha=0.2, loc='upper right')
ax2.set_ylim(bottom=line_min, top=line_max)
# plt.xticks(rotation=45)
ax2.tick_params(axis='x', labelrotation=45 )


ax2.set_ylabel('Magnitude')
# ax2.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2, 0, 0))
ax2.legend(loc='upper center', ncol=2)

plt.show()

