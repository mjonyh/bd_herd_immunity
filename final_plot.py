import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns
sns.set()

import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls

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

files = ['Bangladesh_90', 'Bangladesh_30']
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

### mobility https://www.google.com/covid19/mobility/
df_mobility = pd.read_csv('data/test.csv')
df_mobility = df_mobility[df_mobility['country_region_code']=='BD']
df_mobility = df_mobility[['date','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline']]
df_mobility_mean = df_mobility[['retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline']]
df_mobility['mean'] = df_mobility_mean.mean(axis=1)
# print(df_mobility.head())
df_mobility = convert_numeric(df_mobility, 'date')
# df_mobility.plot(x='date')

data = {'Cases': [ 10.2, 54.6, 28.4, 6.7 ],
        'Deaths': [ 2.86, 11.57, 48.49, 37.08 ],
        'label': ['1-20', '21-40', '41-60', '60+']
        }

df = pd.DataFrame(data)

### Hospitals beds: https://public.tableau.com/profile/masud.parvez7954#!/vizhome/Logistics_15857300704580/facility-1
hospital_beds = 7034

fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)

for i in range(len(files)):

    ### Real data
    df_1 = pd.read_csv('data/'+files[i]+'_1.csv', encoding='UTF-8')
    df_1 = convert_numeric(df_1, 'days')

    ### Simulated data
    df_2 = pd.read_csv('data/'+files[i]+'_2.csv', encoding='UTF-8')
    df_2 = convert_numeric(df_2, 'days_sim')

    ### cases
    if (i<1):
        tag = [11, 'semilogy', 'kx', None, None]
        graph_plot(df_1['days'], df_1['active'], tag, ax1)       # active cases
        tag = [11, 'semilogy', 'mx', None, None]
        graph_plot(df_1['days'], df_1['confirmed'].diff(), tag, ax1)       # growth cases
        tag = [11, 'semilogy', 'rx', None, None]
        graph_plot(df_1['days'], df_1['deaths'], tag, ax1)       # death cases
        ax1.axhline(hospital_beds, color='black', linestyle='dotted')
        ax1.text('2020-03-01', hospital_beds*1.2, 'Covid Beds', size=10)
        # tag = [11, 'semilogy', 'gx', None, None]
        # graph_plot(home_isolation_date, home_isolation, tag)
        # graph_plot(df_isolation['date'], df_isolation['isolation'], tag)
        # plt.axvline('2020-05-25', color='maroon', alpha=0.5, linestyle='--')
        # plt.text('2020-05-26', 1e5, 'Regional\nLockdown', color='maroon', size=10)
        # plt.axvline('2020-05-05', color='green', alpha=0.5, linestyle='--')
        # plt.text('2020-05-06', 1e5, 'Lockdown\nRelaxed', color='green', size=10)
        # plt.axvline('2020-04-09', color='black', alpha=0.5, linestyle='--')
        # plt.text('2020-04-10', 1e5, 'Lockdown\nStarted', size=10)


    ### active cases
    tag = [11, 'semilogy', marker[i], 'A ('+population[i]+')', None]
    graph_plot(df_2['days_sim'], df_2['active_sim'], tag, ax1)

    ### hospital required
    tag = [11, 'semilogy', marker[i+2], 'HR ('+population[i]+')', None]
    graph_plot(df_2['days_sim'], df_2['active_sim']*16/100, tag, ax1)

    ### Death Cases
    tag = [11, 'semilogy', marker[i+4], 'D ('+population[i]+')', 'Number of Cases']
    graph_plot(df_2['days_sim'], df_2['deaths_sim'], tag, ax1)

    # ### susceptible Cases
    # tag = [11, 'semilogy', marker[i+6], 'S ('+population[i]+')', 'Number of Cases']
    # graph_plot(df_2['days_sim'], df_2['susceptible_sim'], tag)

    ### growth cases
    tag = [11, 'semilogy', marker[i+8], 'G ('+population[i]+')', 'Number of Cases']
    graph_plot(df_2['days_sim'], df_2['confirmed_sim'].diff(), tag, ax1)
    g_max = df_2['confirmed_sim'].diff().max()
    ax1.axhline(g_max, color='m', linestyle=':')
    ax1.text('2020-03-01', g_max*1.2, '$G_{max}$: '+str(int(g_max/1e3))+'K', size=10)

    # print(int(df_2['confirmed_sim'][df_2['days_sim']=='2020-09-01'].values[0]))

    ax1.set_ylim(bottom=1, top=1e6)
    ax1.set_xlim(xmin='2020-03-01', xmax='2021-04-01')
    # plt.xlim(xmin='2020-03-01')
    ax1.legend(ncol=2, fancybox=True, framealpha=0.2)

    ### reproduction numbers
    if (i<1):

        ### R0
        # plt.axhline(final_r0, color='black', linestyle='-', label='$R_0$')
        # plt.fill_between(df_2['days_sim'], final_r0+0.25, final_r0-0.25, color='k', alpha=0.2)
        # plt.text('2020-11-15', 3.8, '$R_0$', rotation=0, color='k')
        # plt.axvline('2020-05-25', color='maroon', alpha=0.5, label = '$L_1$')
        # plt.axvline('2020-05-05', color='green', alpha=0.5, label = '$L_2$')
        # plt.axvline('2020-04-09', color='black', alpha=0.5, label = '$L_3$')

        # % Lockdown: 26th March, 2020: https://bdnews24.com/bangladesh/2020/03/25/bangladesh-in-virtual-lockdown-as-coronavirus-fight-flares
        # % Reopen Garments: 26 April, 2020: https://bdnews24.com/bangladesh/2020/04/26/hundreds-of-clothing-factory-workers-dash-for-dhaka-again-in-dark-hours-amid-lockdown
        # % Shopping Mall reopen: 10th May, 2020: https://bdnews24.com/business/2020/05/04/shopping-malls-to-reopen-ahead-of-eid
        # % Restarting economy: 30th May, 2020: https://bdnews24.com/bangladesh/2020/05/29/bangladesh-set-to-step-into-coronavirus-new-normal-with-a-lot-at-stake

        ### mobility
        tag = [12, 'plot', 'k-', 'Negative Mobility (x10)', None]
        graph_plot(df_mobility['date'], -df_mobility['mean']/10, tag, ax2)
        tag = [12, 'plot', 'k.', None, None]
        graph_plot(df_mobility['date'], -df_mobility['mean']/10, tag, ax2)

        ### growth
        tag = [12, 'plot', 'b-', 'Daily Growth (x1K)', None]
        graph_plot(df_1['days'], df_1['confirmed'].diff()/1000, tag, ax2)
        tag = [12, 'plot', 'b.', None, None]
        graph_plot(df_1['days'], df_1['confirmed'].diff()/1000, tag, ax2)

        ### doublingtimes
        tag = [12, 'plot', 'g-', 'Doubling Time (x10)', None]
        graph_plot(df_doublings_1['date'], df_doublings_1['doublingtimes']/10, tag, ax2)
        tag = [12, 'plot', 'g.', None, None]
        graph_plot(df_doublings_1['date'], df_doublings_1['doublingtimes']/10, tag, ax2)

        ### lock down
        text_position = 4.5
        line_max = 6
        ax2.vlines(x='2020-03-26', ymin=0, ymax=line_max, color='black', alpha=0.5)
        ax2.text('2020-03-27', text_position, '$L_1$\nC: 44\nD: 5', size=10)
        ax2.vlines('2020-04-26', ymin=0, ymax=line_max, color='k', alpha=0.5)
        ax2.text('2020-04-27', text_position, '$L_2$\nC: 5416\nD: 145', color='black', size=10)
        ax2.vlines('2020-05-10', color='k', alpha=0.5, ymin=0, ymax=line_max)
        ax2.text('2020-05-11', text_position, '$L_3$\nC: 14657\nD: 228', color='black', size=10)
        ax2.vlines('2020-05-30', color='k', alpha=0.5, ymin=0, ymax=line_max)
        ax2.text('2020-05-31', text_position, '$L_4$\nC: 44608\nD: 610', color='black', size=10)

        ### Rt
        tag = [12, 'plot', 'r-', '$R_t$', None]
        graph_plot(df_rt_1['Date'], df_rt_1['ML'], tag, ax2)       # active cases
        tag = [12, 'plot', 'r.', None, None]
        graph_plot(df_rt_1['Date'], df_rt_1['ML'], tag, ax2)       # active cases
        test = ax2.fill_between(df_rt_1['Date'], df_rt_1['Low_90'], df_rt_1['High_90'], alpha=0.5, color='r')

        ax2.set_ylabel('Magnitude')
    ax2.set_xlim(xmin='2020-03-15', xmax='2020-08-01')
    ax2.legend(ncol=2, framealpha=0.2, loc='upper right')
    ax2.set_ylim(bottom=0, top=7.5)

    if(i==0):
        last_data_date = (df_1['days'].iloc[len(df_1)-1])
        last_data_index = len(df_1)-1

        save_df = df_2[['days_sim', 'confirmed_sim', 'recovered_sim', 'deaths_sim']].iloc[last_data_index:last_data_index+14]
        save_df.columns = ['Date', 'Confirmed Cases', 'Recovered Cases', 'Deaths']
        save_df[['Confirmed Cases', 'Recovered Cases', 'Deaths']] = save_df[['Confirmed Cases', 'Recovered Cases', 'Deaths']].astype(int)
        save_df['Rt'] = df_rt_2['ML'].iloc[last_data_index:last_data_index+14]
        save_df['DT'] = df_doublings_2['doublingtimes'].iloc[last_data_index:last_data_index+14].round()

        save_df.to_csv('data/forcasting.csv')

# plt.show()

import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls
import plotly.express as px
import plotly.graph_objects as go

username='mjonyh-phy'
api_key='BeJs4fGTnPuNtmWNWZ5C'
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

figure_1 = go.Figure(data=[
        go.Scatter(
            x=df_2['days_sim'],
            y=df_2['confirmed_sim'],
            name='Estimated Confirm Cases'
            ),
        go.Scatter(
            x=df_1['days'],
            y=df_1['confirmed'],
            mode='markers',
            # marker=dict(color="Black"),
            name='Real Confirmed Cases'
            ),

        go.Scatter(
            x=df_2['days_sim'],
            y=df_2['recovered_sim'],
            name='Estimated Recovered Cases'
            ),
        go.Scatter(
            x=df_1['days'],
            y=df_1['recovered'],
            mode='markers',
            name='Real Recovered Cases'
            ),

        go.Scatter(
            x=df_2['days_sim'],
            y=df_2['deaths_sim'],
            name='Estimated Deaths Cases'
            ),
        go.Scatter(
            x=df_1['days'],
            y=df_1['deaths'],
            mode='markers',
            name='Real Deaths Cases'
            ),

        go.Scatter(
            x=df_2['days_sim'],
            y=df_2['active_sim'],
            name='Estimated Active Cases'
            ),
        go.Scatter(
            x=df_1['days'],
            y=df_1['active'],
            mode='markers',
            name='Real Active Cases'
            )
        ])

figure_1.update_layout(
        yaxis_type='log',
        )

figure_1.update_xaxes(range=['2020-04-01', '2021-01-01'])

figure_2 = go.Figure(data=[
    go.Scatter(
        x=df_rt_2['Date'],
        y=df_rt_2['ML'],
        name='Predicted Rt'
        ),
    go.Scatter(
        x=df_doublings_2['date'],
        y=df_doublings_2['doublingtimes']/10,
        name='Predicted Doubling Times (x10)'
        ),

    go.Scatter(
        x=df_rt_2['Date'],
        y=df_rt_2['High_90'],
        fill=None,
        name='Predicted High 90'
        ),

    go.Scatter(
        x=df_rt_2['Date'],
        y=df_rt_2['Low_90'],
        fill='tonexty',
        name='Predicted Low 90'
        ),
    go.Scatter(
        x=df_rt_1['Date'],
        y=df_rt_1['ML'],
        name='Real Rt'
        ),
    go.Scatter(
        x=df_doublings_1['date'],
        y=df_doublings_1['doublingtimes']/10,
        name='Real Doubling Times (x10)'
        ),

    go.Scatter(
        x=df_rt_1['Date'],
        y=df_rt_1['High_90'],
        fill=None,
        name='Real High 90'
        ),

    go.Scatter(
        x=df_rt_1['Date'],
        y=df_rt_1['Low_90'],
        fill='tonexty',
        name='Real Low 90'
        ),
    go.Scatter(
        x=df_mobility['date'],
        y=-df_mobility['mean']/10,
        name='Drop in Mobility (x10)'
        )
    ])

figure_2.update_xaxes(range=['2020-04-01', '2021-01-01'])
figure_2.update_yaxes(range=[-0.1, 8.0])
figure_2.update_layout(shapes=[
    dict(
      type= 'line',
      xref= 'x', x0= '2020-04-01', x1= '2021-01-01',
      yref= 'y', y0= 1, y1= 1,
    )
])

py.plot(figure_1, filename='figure 1')
py.plot(figure_2, filename='figure 2')
