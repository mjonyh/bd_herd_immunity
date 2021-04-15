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
mpl.rcParams['figure.subplot.bottom'] = 0.055

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
mpl.rcParams['figure.subplot.top'] = 0.92

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

# files = ['Bangladesh_90', 'Bangladesh_30']
files = ['Bangladesh_90']
population = ['53M', '18M', 'HR (53M)', 'HR (18M)']
marker = ['k-', 'k-.', 'g-', 'g-.', 'r-', 'r-.', 'b-', 'b-.', 'm-', 'm-.']
linestyle = ['-', '-.']

plot_data_semilog = ['confirmed', 'recovered', 'deaths', 'active']
plot_data_label_semilog = ['Confirmed', 'Recovered', 'Death', 'Active']

plot_data_plot = ['cfr_recovered', 'cfr_confirmed']
plot_data_label_plot = ['Case Fatality Rate (in %)', 'Case Fatality Rate (in %)']

plot_data_plot_sim = ['r0', 're']
plot_data_label_plot_sim = ['$R_0$', 'Reproduction Number', '$R_e$']
reproduction_legend = ['$R_0$', '$R_e$']

df_rt_1 = pd.read_csv('data/rt_1.csv')
df_rt_1 = convert_numeric(df_rt_1, 'Date')

df_rt_2 = pd.read_csv('data/rt_2.csv')
df_rt_2 = convert_numeric(df_rt_2, 'Date')
# print(df_rt)

df_doublings_1 = pd.read_csv('data/doublingtimes_1.csv')
df_doublings_1 = convert_numeric(df_doublings_1, 'date')

df_doublings_2 = pd.read_csv('data/doublingtimes_2.csv')
df_doublings_2 = convert_numeric(df_doublings_2, 'date')

df_isolation = pd.read_csv('data/isolation.csv')
df_isolation = convert_numeric(df_isolation, 'date')

# df_owid = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
df_owid = pd.read_csv('data/2021-04-14_owid.csv')
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

data = {'Cases': [ 10.2, 54.6, 28.4, 6.7 ],
        'Deaths': [ 2.86, 11.57, 48.49, 37.08 ],
        'label': ['1-20', '21-40', '41-60', '60+']
        }

df = pd.DataFrame(data)

### Hospitals beds: https://public.tableau.com/profile/masud.parvez7954#!/vizhome/Logistics_15857300704580/facility-1
hospital_beds = 7034

fig2, ax2 = plt.subplots(2, 1, figsize=(6.4, 8), sharex=True)

xmax_limit = '2021-04-01'

for i in range(len(files)):

    ### Real data
    df_1 = pd.read_csv('data/'+files[i]+'_1.csv', encoding='UTF-8')
    df_1 = convert_numeric(df_1, 'days')

    ### Simulated data
    df_2 = pd.read_csv('data/'+files[i]+'_2.csv', encoding='UTF-8')
    df_2 = convert_numeric(df_2, 'days_sim')

    ### reproduction numbers
    if (i<1):

        color = 'tab:red'

        ### mobility
        tag = [12, 'plot', 'k-', 'Drops in Mobility (x10)', None]
        graph_plot(df_mobility['date'], -df_mobility['mean']/10, tag, ax2[0])
        tag = [12, 'plot', 'k.', None, None]
        graph_plot(df_mobility['date'], -df_mobility['mean']/10, tag, ax2[0])
        # ax2.plot(df_mobility['date'], -df_mobility['mean']/10, color=color)
        # l1, = ax2.plot(df_mobility['date'], -df_mobility['mean']/10, color=color, marker='^', label='Drops in Mobility (x10)')

        ### growth
        tag = [12, 'plot', 'b-', 'Daily Cases (x1K)', None]
        graph_plot(df_1['days'], df_1['confirmed'].diff()/1000, tag, ax2[0])
        tag = [12, 'plot', 'b.', None, None]
        graph_plot(df_1['days'], df_1['confirmed'].diff()/1000, tag, ax2[0])
        tag = [12, 'plot', 'b--', None, None]
        graph_plot(df_2['days_sim'], df_2['confirmed_sim'].diff()/1000, tag, ax2[0])

        ### Test Positive Rate
        tag = [12, 'plot', 'g-', 'Test Positive Rate (x10 in %)', None]
        graph_plot(df_owid['date'], df_owid['positive_rate']*10, tag, ax2[0])
        tag = [12, 'plot', 'g.', None, None]
        graph_plot(df_owid['date'], df_owid['positive_rate']*10, tag, ax2[0])


        ### doublingtimes
        # tag = [12, 'plot', 'g-', 'Doubling Time (x100)', None]
        # graph_plot(df_doublings_1['date'], df_doublings_1['doublingtimes']/100, tag, ax2)
        # tag = [12, 'plot', 'g.', None, None]
        # graph_plot(df_doublings_1['date'], df_doublings_1['doublingtimes']/100, tag, ax2)

        ### lock down
        text_position = 8.5
        text_position_down = 4.5
        line_max = 10
        line_min = -2
        ### First locked down
        ax2[0].vlines('2020-03-26', ymin=line_min, ymax=line_max, color='k', alpha=0.5)
        ax2[0].text('2020-03-27', text_position, '$L_1$', size=10)
        ### Reopen Garments Factory
        ax2[0].vlines('2020-04-26', ymin=line_min, ymax=line_max, color='k', alpha=0.5)
        ax2[0].text('2020-04-27', text_position, '$L_2$', color='black', size=10)
        ### Shopping Mall reopened (Eid-ul-Fitr)
        ax2[0].vlines('2020-05-10', color='k', alpha=0.5, ymin=line_min, ymax=line_max)
        ax2[0].text('2020-05-11', text_position, '$L_3$', color='black', size=10)
        ### Restarting economy
        ax2[0].vlines('2020-05-30', color='k', alpha=0.5, ymin=line_min, ymax=line_max)
        ax2[0].text('2020-05-31', text_position, '$L_4$', color='black', size=10)
        ### Eid ul Adha
        ax2[0].vlines('2020-07-31', color='k', alpha=0.5, ymin=line_min, ymax=line_max)
        ax2[0].text('2020-08-01', text_position, '$L_5$', color='black', size=10)
        ### durga puja
        ax2[0].vlines('2020-10-26', color='k', alpha=0.5, ymin=line_min, ymax=line_max)
        ax2[0].text('2020-10-27', text_position, '$L_6$', color='black', size=10)
        ### No more free test
        ax2[0].vlines('2020-06-30', color='r', alpha=0.5, ymin=line_min, ymax=line_max)
        ax2[0].text('2020-07-01', text_position, '$G_1$', color='r', size=10)
        ### Winter
        ax2[0].vlines('2020-11-15', color='r', alpha=0.5, ymin=line_min, ymax=line_max)
        ax2[0].text('2020-11-16', text_position, 'Winter', color='r', size=10)
        ### 21st February
        ax2[0].vlines('2021-02-21', color='k', alpha=0.5, ymin=line_min, ymax=line_max)
        ax2[0].text('2021-02-22', text_position, '$L_7$', color='k', size=10)
        ### 26th march
        ax2[0].vlines('2021-03-26', color='k', alpha=0.5, ymin=line_min, ymax=line_max)
        ax2[0].text('2021-03-27', text_position, '$L_8$', color='k', size=10)
        ### locked down
        ax2[0].vlines('2021-04-05', color='k', alpha=0.5, ymin=line_min, ymax=line_max)
        ax2[0].text('2021-04-06', text_position, '$L_9$', color='k', size=10)
        ### complete locked down
        ax2[0].vlines('2021-04-14', color='k', alpha=0.5, ymin=line_min, ymax=line_max)
        ax2[0].text('2021-04-15', text_position, '$L_{10}$', color='k', size=10)


        ### Rt
        tag = [12, 'plot', 'r-', '$R_t$', None]
        graph_plot(df_rt_1['Date'], df_rt_1['ML'], tag, ax2[0])       # active cases
        # ax2.plot(df_rt_1['Date'], df_rt_1['ML'], color=color)
        # l2, = ax2.plot(df_rt_1['Date'], df_rt_1['ML'], color=color, marker='.', label='$R_t$')
        tag = [12, 'plot', 'r.', None, None]
        graph_plot(df_rt_1['Date'], df_rt_1['ML'], tag, ax2[0])       # active cases
        annot_max(df_rt_1['Date'], df_rt_1['ML'], ax2[0], 'r')

        test = ax2[0].fill_between(df_rt_1['Date'], df_rt_1['Low_90'], df_rt_1['High_90'], alpha=0.5, color='r')

        # ax2.set_ylabel('Magnitude', color=color)
        # ax2.tick_params(axis='y',labelcolor=color)

        df = pd.read_csv('data/owid-covid-data.csv')
        df = df[df['location']=='Bangladesh']
        df['date'] = pd.to_datetime(df['date'])

        # ax3 = ax2.twinx()
        # color = 'tab:blue'

        # ax3.plot(df['date'], df['positive_rate']*100, color=color)
        # l3, = ax3.plot(df['date'], df['positive_rate']*100, color=color, marker='.', label='Positive Test Rate (in %)')
        # ax3.set_ylim(0, 40)
        # ax3.set_ylabel("Positive Test Rate (in %)",color=color)
        # ax3.tick_params(axis='y',labelcolor=color)

        # ax3.legend(handles=[l1, l2, l3], loc=2, ncol=2)
        # fig2.tight_layout()

    ax2[0].set_xlim(xmin='2020-03-15', xmax=xmax_limit)
    # ax2.legend(ncol=2, framealpha=0.2, loc='upper right')
    ax2[0].set_ylim(bottom=line_min, top=line_max)
    # plt.xticks(rotation=45)
    ax2[0].tick_params(axis='x', labelrotation=45 )

df = pd.read_csv('data/zone_risk.csv')
df.date = pd.to_datetime(df.date)

df.plot(x='date', y=['green', 'yellow', 'orange', 'red'], ax= ax2[1],
        color = ['#54b45f', '#ecd424', '#f88c51', '#c01a27'],
        label = ['Trivial', 'Community Spread', 'Accelerated Spread', 'Tipping Point'],
        ylabel='Number of Districts'
        )

df_forecast = pd.read_csv('data/zone_risk_forecast.csv')
df_forecast.date = pd.to_datetime(df_forecast.date)

df_forecast.plot(x='date', y=['green', 'yellow', 'orange', 'red'], ax= ax2[1],
        color = ['#54b45f', '#ecd424', '#f88c51', '#c01a27'],
        style='--',
        label = ['Trivial (Forecast)', 'Community Spread (Forecast)', 'Accelerated Spread (Forecast)', 'Tipping Point (Forecast)'],
        ylabel='Number of Districts',
        )

# tag = [12, 'plot', '#ecd424--', None, None]
# graph_plot(df_2['days_sim'], df_2['confirmed_sim'].diff()/1000, tag, ax2[0])
# tag = [12, 'plot', 'b--', None, None]
# graph_plot(df_2['days_sim'], df_2['confirmed_sim'].diff()/1000, tag, ax2[0])
# tag = [12, 'plot', 'b--', None, None]
# graph_plot(df_2['days_sim'], df_2['confirmed_sim'].diff()/1000, tag, ax2[0])
# tag = [12, 'plot', 'b--', None, None]
# graph_plot(df_2['days_sim'], df_2['confirmed_sim'].diff()/1000, tag, ax2[0])

ax2[0].set_ylabel('Magnitude')
# ax2[0].set_yscale('log')
ax2[0].legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2, 0, 0))
ax2[1].legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3, 0, 0))
ax2[1].set_xlim(xmin='2020-03-15')
# ax.legend(loc=9, ncol=2, bbox_to_anchor=(0.5, 1.2, 0, 0))

# first_date, last_date = df.date.iloc[[0,-1]]
# print(first_date, last_date)

# df_1['daily_cases'] = df_1['confirmed'].diff()
# df_1 = df_1[(df_1.days >=first_date)&(df_1.days <=last_date)]

# print(len(df.date), len(df_1.days))

# X = df[['green', 'yellow', 'orange', 'red']]
# Y = df_1['daily_cases']

# from sklearn import linear_model
# # regr = linear_model.LinearRegression()
# regr = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
# regr.fit(X, Y)

# # print(regr.intercept_, regr.coef_)

# pred_X = df_forecast[['green', 'yellow', 'orange', 'red']]

# pred_x = [[16,45,2,1]]
# predictions = regr.predict(pred_x)

# # MSE = np.mean((Y - regr.predict(X).T)**2)
# # var_est = MSE * np.diag(np.linalg.pinv(np.dot(X.T,X)))
# # SE_est = np.sqrt(var_est)
# # se_prediction = regr.predict([SE_est])

# # print(predictions, SE_est, se_prediction)
# print(predictions, regr.score(pred_x, predictions))

plt.show()

