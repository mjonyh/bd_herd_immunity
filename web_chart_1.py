import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns
sns.set()


def convert_numeric(df, date_column):
    cols=[i for i in df.columns if i not in [date_column]]
    for col in cols:
        df[col]=pd.to_numeric(df[col])
    df[date_column] = pd.to_datetime(df[date_column])
    return df


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
# population = ['53M', '18M', 'HR (53M)', 'HR (18M)']
marker = ['k-', 'k-.', 'g-', 'g-.', 'r-', 'r-.', 'b-', 'b-.', 'm-', 'm-.']
linestyle = ['-', '-.']

### Hospitals beds: https://public.tableau.com/profile/masud.parvez7954#!/vizhome/Logistics_15857300704580/facility-1
hospital_beds = 7034

fig1, ax1 = plt.subplots(1)
# fig2, ax2 = plt.subplots(1)

xmax_limit = '2021-05-01'

### Real data
df_1 = pd.read_csv('data/Bangladesh_90_1.csv', encoding='UTF-8')
df_1 = convert_numeric(df_1, 'days')

### Simulated data
df_2 = pd.read_csv('data/Bangladesh_90_2.csv', encoding='UTF-8')
df_2 = convert_numeric(df_2, 'days_sim')

### cases
# if (i<1):
tag = [11, 'semilogy', 'k.', None, None]
graph_plot(df_1['days'], df_1['active'], tag, ax1)       # active cases
tag = [11, 'semilogy', 'm.', None, None]
graph_plot(df_1['days'], df_1['confirmed'].diff(), tag, ax1)       # growth cases
tag = [11, 'semilogy', 'b.', None, None]
graph_plot(df_1['days'], df_1['confirmed'], tag, ax1)       # confirmed cases
tag = [11, 'semilogy', 'r.', None, None]
graph_plot(df_1['days'], df_1['deaths'], tag, ax1)       # death cases

### active cases
tag = [11, 'semilogy', marker[0], 'Active Cases', None]
graph_plot(df_2['days_sim'], df_2['active_sim'], tag, ax1)

### hospital required
tag = [11, 'semilogy', marker[2], 'Hospital Required', None]
graph_plot(df_2['days_sim'], df_2['active_sim']*16/100, tag, ax1)

### Death Cases
tag = [11, 'semilogy', marker[4], 'Deaths', 'Number of Cases']
graph_plot(df_2['days_sim'], df_2['deaths_sim'], tag, ax1)

### cummulative cases
tag = [11, 'semilogy', marker[6], 'Total Cases', 'Number of Cases']
graph_plot(df_2['days_sim'], df_2['confirmed_sim'], tag, ax1)

### daily cases
tag = [11, 'semilogy', marker[8], 'Daily Cases', 'Number of Cases']
graph_plot(df_2['days_sim'], df_2['confirmed_sim'].diff(), tag, ax1)

ax1.set_ylim(bottom=1, top=1e6)
ax1.set_xlim(xmin='2020-03-01', xmax=xmax_limit)
ax1.legend(ncol=2, fancybox=True, framealpha=0.2)

plt.show()

