#!/usr/bin/env python

from helper import *

import geopandas as gpd
from matplotlib.colors import ListedColormap
import matplotlib as mpl

mpl.rcParams['figure.subplot.left'] = 0
mpl.rcParams['figure.subplot.right'] = 1
mpl.rcParams['figure.subplot.bottom'] = 0
mpl.rcParams['figure.subplot.top'] = 0.95

### https://www.worldbank.org/en/data/interactive/2016/11/10/bangladesh-poverty-maps
filepath = 'data/BD_District_WGS84.shp'

map_df = gpd.read_file(filepath)
# print(map_df.columns) Index(['COMMENTS', 'DIV_NAME', 'DIST_NAME', 'Div_BBS', 'Dist_BBS', 'geometry'], dtype='object')
map_df['district_index'] = map_df['DIST_NAME'].str.slice(stop=2)
map_df['risk'] = np.nan

replace_district_name = {
        'CHITTAGONG': 'CHATTOGRAM' ,
        'COX\'S BAZAR': 'COXS BAZAR' ,
        'NETROKONA': 'NETRAKONA' ,
        'CHAPAI': 'CHAPAINABABGANJ' ,
        'BOGRA': 'BOGURA' ,
        'JESSORE': 'JASHORE' ,
        'MAULVIBAZAR': 'MOULVIBAZAR' ,
        'JHENAIDAHA': 'JHENAIDAH' ,
        'COMILLA': 'CUMILLA' ,
        'BARISAL': 'BARISHAL'
        }

map_df = map_df.replace({'DIST_NAME': replace_district_name})
# print(map_df.DIST_NAME.sort_values().unique())


# print(map_df.columns)

# ss.exit()

bd_population = df_owid['population'].iloc[-1]

# cases_type = 'combined'
df_risk = pd.read_csv('data/zone_risk_value.csv')

this_week = datadate
last_week = pd.to_datetime(datadate) - timedelta(days=7)
last_week = last_week.strftime('%Y-%m-%d')
predicted = pd.to_datetime(datadate) + timedelta(days=7)
predicted = predicted.strftime('%Y-%m-%d')

weeks = [last_week, this_week, predicted]
titles = ['Last Week', 'This Week', 'Next Week (Predicted)']

weeks = [this_week]
titles = ['This Week']

for district in districts:
    iso13 = df_population.loc[df_population['Name']==district.capitalize(), 'Abbr.'].values[0]
    map_df.loc[map_df['DIST_NAME'] == district.upper(), 'district_index'] = iso13

fig, axs = plt.subplots(1, len(weeks), figsize=(4*len(weeks), 6))
variable = 'risk'
i = 0
for week in weeks:
    for district in districts:
        map_df.loc[map_df['DIST_NAME'] == district.upper(), 'risk'] = df_risk.loc[df_risk.district == district, week].values[0]

    print(week, map_df[['DIST_NAME', 'risk']])
    if (i == len(weeks)-1):
        condition = True
    else:
        condition = False

    vmax = map_df[variable].max()
    if(vmax > 24):
        colors = ['#54b45f', '#ecd424', '#f88c51', '#c01a27']
        labels = ['Trivial', 'Community Spread', 'Accelerated Spread', 'Tipping Point']
        bins = [1, 9, 24]
        # test_k_means(merged[variable], axs1[i])
    elif(vmax > 9):
        colors = ['#54b45f', '#ecd424', '#f88c51']
        labels = ['Trivial', 'Community Spread', 'Accelerated Spread']
        bins = [1, 9]
    elif(vmax > 1):
        colors = ['#54b45f', '#ecd424']
        labels = ['Trivial', 'Community Spread']
        bins = [1]
    else:
        colors = ['#54b45f']
        labels = ['Trivial']
        bins = [1]

    map_df.plot(column=variable, linewidth=0.3, ax=axs, edgecolor='0.8',
            cmap=ListedColormap(colors),
            legend_kwds={'loc': 'upper right', 'ncol':1, 'fontsize':8, 'labels':labels, 'title':'Risk'},
            legend=condition,
            scheme='user_defined',
            # missing_kwds={
            #     "color": "lightgrey",
            #     # "edgecolor": "red",
            #     "hatch": "///",
            #     "label": "Missing values",
            #     },
            classification_kwds={'bins':bins}
        )

    axs.set_axis_off()
    # axs[i].set_title('Risk on '+titles[i])
    axs.set_title('Risk on '+weeks[i])
    # create an annotation for the data source
    for index,row in map_df.iterrows():
        xy=row['geometry'].centroid.coords[:]
        xytext=row['geometry'].centroid.coords[:]
        axs.annotate(row['district_index'],xy=xy[0], xytext=xytext[0],  horizontalalignment='center',verticalalignment='center', fontsize=7)


    # print(week)
    start_date = pd.to_datetime(week) - timedelta(days=6)
    start_date = start_date.strftime('%Y-%m-%d')
    # print(start_date, week)
    test_rate_df = df_owid[(df_owid.date >= start_date) & (df_owid.date <= week)]
    total_test = test_rate_df.new_tests.sum()
    test_rate = total_test*1000/bd_population
    # print(test_rate)

    test_label_tr = 'Covid Testing Rate (in 1K population per week): {:.2f}%'.format(test_rate)
    axs.text(88, 20.5, test_label_tr, fontsize=10)
    test_label_rt = 'Reproduction Rate, $R_t$: {:.2f}'.format(test_rate_df[test_rate_df.date == week]['reproduction_rate'].iloc[-1])
    axs.text(88, 20.9, test_label_rt, fontsize=10)
    test_label_tpr = 'Test Positive Rate (Cases per 100 Test): {:.2f}%'.format(test_rate_df[test_rate_df.date == week]['positive_rate'].iloc[-1]*100)
    axs.text(88, 20.7, test_label_tpr, fontsize=10)
    test_label_cases = 'Total Cases: {:.0f}'.format(test_rate_df[test_rate_df.date == week]['total_cases'].iloc[-1])
    axs.text(88, 21.1, test_label_cases, fontsize=10)

plt.show()
