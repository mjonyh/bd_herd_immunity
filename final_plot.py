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


def graph_plot(x, y, tag):
    plt.figure(tag[0])

    if(tag[1]=='plot'):
        ax = plt.plot(x, y, tag[2], label=tag[3])
    elif(tag[1]=='semilogy'):
        ax = plt.semilogy(x, y, tag[2], label=tag[3])


    if(tag[3]!=None):
        plt.legend()

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

df_rt = pd.read_csv('data/rt.csv')
df_rt = convert_numeric(df_rt, 'Date')
# print(df_rt)

df_doublings = pd.read_csv('data/doublingtimes.csv')
df_doublings = convert_numeric(df_doublings, 'date')

df_isolation = pd.read_csv('data/isolation.csv')
df_isolation = convert_numeric(df_isolation, 'date')

### mobility https://www.google.com/covid19/mobility/
df_mobility = pd.read_csv('data/mobility.csv')
df_mobility = df_mobility[df_mobility['country_region_code']=='BD']
df_mobility = df_mobility[['date','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline']]
df_mobility_mean = df_mobility[['retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline']]
df_mobility['mean'] = df_mobility_mean.mean(axis=1)
# print(df_mobility.head())
df_mobility = convert_numeric(df_mobility, 'date')
# df_mobility.plot(x='date')

data = {'Cases': [ 10.2, 54.6, 28.4, 6.7 ],
        'Deaths': [ 2.86, 11.57, 48.49, 37.08 ],
        'label': ['1-20', '21-40', '41-60', '60+']
        }

df = pd.DataFrame(data)

# fig, ax1 = plt.subplots()

### Hospitals beds: https://public.tableau.com/profile/masud.parvez7954#!/vizhome/Logistics_15857300704580/facility-1
hospital_beds = 7034
# home_isolation = pd.to_numeric([6021, 6240, 6498, 6754, 6946, 7162, 7399, 7552, 7893, 8243, 8764, 9012, 9340, 9758, 10026, 10302, 10752, 11026, 11474, 11915, 12190, 12467, 12927, 13158, 13429, 13800])
# home_isolation_date = pd.to_datetime(['2020-06-01', '2020-06-02', '2020-06-03', '2020-06-04', '2020-06-05', '2020-06-06', '2020-06-06', '2020-06-08', '2020-06-09', '2020-06-10', '2020-06-11', '2020-06-12', '2020-06-13', '2020-06-14', '2020-06-15', '2020-06-16', '2020-06-17', '2020-06-18', '2020-06-19', '2020-06-20', '2020-06-21', '2020-06-22', '2020-06-23', '2020-06-24', '2020-06-25', '2020-06-26'])

for i in range(len(files)):

    ### Real data
    df_1 = pd.read_csv('data/'+files[i]+'_1.csv', encoding='UTF-8')
    df_1 = convert_numeric(df_1, 'days')

    ### Simulated data
    df_2 = pd.read_csv('data/'+files[i]+'_2.csv', encoding='UTF-8')
    df_2 = convert_numeric(df_2, 'days_sim')

    # ### cases
    # if (i<1):
    #     tag = [11, 'semilogy', 'kx', None, None]
    #     graph_plot(df_1['days'], df_1['active'], tag)       # active cases
    #     tag = [11, 'semilogy', 'mx', None, None]
    #     graph_plot(df_1['days'], df_1['confirmed'].diff(), tag)       # growth cases
    #     tag = [11, 'semilogy', 'rx', None, None]
    #     graph_plot(df_1['days'], df_1['deaths'], tag)       # death cases
    #     plt.axhline(hospital_beds, color='black', linestyle='dotted')
    #     plt.text('2020-03-01', hospital_beds*1.2, 'Covid Beds', size=10)
    #     # tag = [11, 'semilogy', 'gx', None, None]
    #     # graph_plot(home_isolation_date, home_isolation, tag)
    #     # graph_plot(df_isolation['date'], df_isolation['isolation'], tag)
    #     # plt.axvline('2020-05-25', color='maroon', alpha=0.5, linestyle='--')
    #     # plt.text('2020-05-26', 1e5, 'Regional\nLockdown', color='maroon', size=10)
    #     # plt.axvline('2020-05-05', color='green', alpha=0.5, linestyle='--')
    #     # plt.text('2020-05-06', 1e5, 'Lockdown\nRelaxed', color='green', size=10)
    #     # plt.axvline('2020-04-09', color='black', alpha=0.5, linestyle='--')
    #     # plt.text('2020-04-10', 1e5, 'Lockdown\nStarted', size=10)


    # ### active cases
    # tag = [11, 'semilogy', marker[i], 'A ('+population[i]+')', None]
    # graph_plot(df_2['days_sim'], df_2['active_sim'], tag)

    # ### hospital required
    # tag = [11, 'semilogy', marker[i+2], 'HR ('+population[i]+')', None]
    # graph_plot(df_2['days_sim'], df_2['active_sim']*16/100, tag)

    # ### Death Cases
    # tag = [11, 'semilogy', marker[i+4], 'D ('+population[i]+')', 'Number of Cases']
    # graph_plot(df_2['days_sim'], df_2['deaths_sim'], tag)

    # # ### susceptible Cases
    # # tag = [11, 'semilogy', marker[i+6], 'S ('+population[i]+')', 'Number of Cases']
    # # graph_plot(df_2['days_sim'], df_2['susceptible_sim'], tag)

    # ### growth cases
    # tag = [11, 'semilogy', marker[i+8], 'G ('+population[i]+')', 'Number of Cases']
    # graph_plot(df_2['days_sim'], df_2['confirmed_sim'].diff(), tag)
    # g_max = df_2['confirmed_sim'].diff().max()
    # plt.axhline(g_max, color='m', linestyle=':')
    # plt.text('2020-03-01', g_max*1.2, '$G_{max}$: '+str(int(g_max/1e3))+'K', size=10)

    # # print(int(df_2['confirmed_sim'][df_2['days_sim']=='2020-09-01'].values[0]))

    # plt.ylim(bottom=1, top=1e6)
    # plt.xlim(xmin='2020-03-01', xmax='2021-04-01')
    # # plt.xlim(xmin='2020-03-01')
    # plt.legend(ncol=2, fancybox=True, framealpha=0.2)

    ### Reproduction number
    final_r0 = df_2['r0'][50:len(df_2['r0'])-1].mean()
    print('R0 = ', df_2['r0'][50:len(df_2['r0'])-1].mean())
    print('beta = ', df_2['beta'][50:len(df_2['r0'])-1].mean())
    print('gamma = ', df_2['gamma'][50:len(df_2['r0'])-1].mean())
    print('mu = ', df_2['mu'][50:len(df_2['r0'])-1].mean())

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
        graph_plot(df_mobility['date'], -df_mobility['mean']/10, tag)
        tag = [12, 'plot', 'k.', None, None]
        graph_plot(df_mobility['date'], -df_mobility['mean']/10, tag)

        ### growth
        tag = [12, 'plot', 'b-', 'Daily Growth (x1K)', None]
        graph_plot(df_1['days'], df_1['confirmed'].diff()/1000, tag)
        tag = [12, 'plot', 'b.', None, None]
        graph_plot(df_1['days'], df_1['confirmed'].diff()/1000, tag)

        ### doublingtimes
        tag = [12, 'plot', 'g-', 'Doubling Time (x10)', None]
        graph_plot(df_doublings['date'], df_doublings['doublingtimes']/10, tag)
        tag = [12, 'plot', 'g.', None, None]
        graph_plot(df_doublings['date'], df_doublings['doublingtimes']/10, tag)

        ### lock down
        text_position = 4.5
        line_max = 6
        plt.vlines(x='2020-03-26', ymin=0, ymax=line_max, color='black', alpha=0.5)
        plt.text('2020-03-27', text_position, '$L_1$\nC: 44\nD: 5', size=10)
        plt.vlines('2020-04-26', ymin=0, ymax=line_max, color='k', alpha=0.5)
        plt.text('2020-04-27', text_position, '$L_2$\nC: 5416\nD: 145', color='black', size=10)
        plt.vlines('2020-05-10', color='k', alpha=0.5, ymin=0, ymax=line_max)
        plt.text('2020-05-11', text_position, '$L_3$\nC: 14657\nD: 228', color='black', size=10)
        plt.vlines('2020-05-30', color='k', alpha=0.5, ymin=0, ymax=line_max)
        plt.text('2020-05-31', text_position, '$L_4$\nC: 44608\nD: 610', color='black', size=10)

        ### Rt
        tag = [12, 'plot', 'r-', '$R_t$', None]
        graph_plot(df_rt['Date'], df_rt['ML'], tag)       # active cases
        tag = [12, 'plot', 'r.', None, None]
        graph_plot(df_rt['Date'], df_rt['ML'], tag)       # active cases
        plt.fill_between(df_rt['Date'], df_rt['Low_90'], df_rt['High_90'], alpha=0.5, color='r')

        plt.ylabel('Magnitude')
    plt.xlim(xmin='2020-03-15', xmax='2020-08-01')
    plt.legend(ncol=2, framealpha=0.2, loc='upper right')
    #     # plt.text('2020-06-10', 2.0, 'Growth (x1K)', rotation=0, color='g')

    # # tag = [12, 'plot', marker[i+4], '$R_e$ ('+population[i]+')', 'Reproduction Number']
    # # graph_plot(df_2['days_sim'], final_r0*df_2['susceptible_sim']/df_2['susceptible_sim'][0], tag) 
    # # plt.fill_between(df_2['days_sim'], final_r0*df_2['susceptible_sim']/df_2['susceptible_sim'][0]+0.25, final_r0*df_2['susceptible_sim']/df_2['susceptible_sim'][0]-0.25, color='r', alpha=0.2)

    plt.ylim(bottom=0, top=7.5)
    # # plt.xlim('2020-03-15', '2020-07-01')
    # # plt.legend(ncol=3, loc='upper right', fancybox=True, framealpha=0.2)
    # plt.legend(ncol=2, framealpha=0.2, loc='upper right')

    # # test_df = df_2[df_2['days_sim']>'2020-05-14']
    # # print(test_df)
    # # print(test_df['r0'].mean(), test_df['r0'].std(ddof=0))

    # # plt.figure(13)
    # # plt.plot(df_1['days'], df_1['confirmed'], '-')
    # # plt.plot(df_1['days'], df_1['confirmed'], 'o')

    # plt.xlim(xmin='2020-03-15', xmax='2020-08-01')
    # plt.ylim(0,7.5)

    # ### CFR
    # if (i>0):
    #     tag = [13, 'plot', 'kx', None, None]
    #     graph_plot(df_1['days'], df_1['cfr_recovered'], tag)       # active cases
    #     tag = [13, 'plot', 'gx', None, None]
    #     graph_plot(df_1['days'], df_1['cfr_confirmed'], tag)       # active cases
    #     tag = [13, 'plot', marker[0], 'CFR$_{adj}$', None]
    #     graph_plot(df_2['days_sim'], df_2['cfr_recovered_sim'], tag)       # active cases
    #     tag = [13, 'plot', marker[i+1], 'CFR', 'Case Fatality Rate (in %)']
    #     graph_plot(df_2['days_sim'], df_2['cfr_confirmed_sim'], tag)       # active cases

    #     for k in range(len(df['label'])):
    #         tag = [13, 'plot', marker[i+2+k], 'Age: '+df['label'][k], 'Case Fatality Rate (in %)']
    #         graph_plot(df_2['days_sim'], df_2['cfr_confirmed_sim']*df['Deaths'][k]/df['Cases'][k], tag)       # active cases

    #     plt.xlim(xmin='2020-05-01', xmax='2021-10-01')
    # plt.ylim(0,15)
    # plt.legend(ncol=3, loc='upper center', fancybox=True, framealpha=0.2)

    # # ### Reproduction number
    # # final_r0 = df_2['r0'][len(df_2['r0'])-1]
    # # # tag = [14, 'plot', marker[i+4], '$R_e$ ('+population[i]+')', 'Reproduction Number']
    # # # graph_plot(df_2['days_sim'], final_r0*df_2['susceptible_sim']/df_2['susceptible_sim'][0], tag)       # active cases
    # # # plt.fill_between(df_2['days_sim'], final_r0*df_2['susceptible_sim']/df_2['susceptible_sim'][0]+0.5, final_r0*df_2['susceptible_sim']/df_2['susceptible_sim'][0]-0.5, color='r', alpha=0.2)
    
    # # if (i<1):
    # #     # tag = [12, 'plot', 'k-', '$R_0$', None]
    # #     # graph_plot(df_2['days_sim'], df_2['r0'], tag)       # active cases
    # #     tag = [14, 'plot', 'k-', '$R_t$', None]
    # #     graph_plot(df_rt['Date'], df_rt['ML'], tag)       # active cases
    # #     tag = [14, 'plot', 'r.', None, None]
    # #     graph_plot(df_rt['Date'], df_rt['ML'], tag)       # active cases
    # #     plt.fill_between(df_rt['Date'], df_rt['Low_90'], df_rt['High_90'], color='black', alpha=0.2)
    # #     # plt.axhline(final_r0, color='black', linestyle='-', label='$R_0$')
    # #     # plt.fill_between(df_2['days_sim'], final_r0+0.5, final_r0-0.5, color='k', alpha=0.2)
    # #     tag = [14, 'plot', 'g.', None, None]
    # #     graph_plot(df_1['days'], df_1['confirmed'].diff()/1000, tag)
    # #     tag = [14, 'plot', 'g-', 'Confirmed (x1K)', None]
    # #     graph_plot(df_1['days'], df_1['confirmed'].diff()/1000, tag)
    # #     plt.xlim(xmin='2020-04-01')
    # #     plt.ylim(0, 5)

    # last_data_index = len(df_1['days'])
    # test_df = df_2[df_2['active_sim'] == df_2['active_sim'].max()]
    # print()
    # print('Peak Confirmed: ', test_df['confirmed_sim'], 'Last Confirmed: ', df_1['confirmed'][last_data_index-1], 'last deaths: ', df_1['deaths'][last_data_index-1], 'last active: ', df_1['active'][last_data_index-1], 'R0: ', final_r0)
    # print(df_1['days'][last_data_index-1], 'Peak Susceptible: ', test_df['susceptible_sim'], 'Peak active: ', test_df['active_sim'])
    # print(test_df)

    # ### for observation ###
    # # # for i in range(last_data_index-10, last_data_index - 1):
    # # #     print('Real: ', df_1['confirmed'][i], 'Simulated: ', df_2['confirmed_sim'][i])
    # # print('Date \t \t Confirmed \t \t Recovered \t \t Deaths')
    # # j = 0
    # # for i in range(last_data_index, last_data_index+14):
    # #     print_confirmed = int(df_2['confirmed_sim'][i])
    # #     print_recovered = int(df_2['recovered_sim'][i])
    # #     print_deaths = int(df_2['deaths_sim'][i])

    # #     print(df_2['days_sim'][i].strftime("%Y-%m-%d"), '\t', print_confirmed, '+/-', int(print_confirmed*(0.5+(j*0.01))/100), '\t', print_recovered, '+/-', int(print_recovered*(0.5+(j*0.01))/100), '\t', print_deaths, '+/-', int(print_deaths*(0.5+(j*0.01))/100))
    # #     j = j+1


    # # test_df = df_2[df_2['active_sim'] == df_2['active_sim'].max()]
    # # print('Peak Confirmed: ', test_df['confirmed_sim'], 'Last Confirmed: ', df_1['confirmed'][last_data_index-1])

plt.show()
