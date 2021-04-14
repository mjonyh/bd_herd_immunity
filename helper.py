import sys as ss
from os import path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from scipy.stats import linregress

### Covid-19 districts data
datadate = datetime.today() - timedelta(days=1)
datadate = datadate.strftime('%Y-%m-%d')
# datadate = '2021-04-10'

def download_a2i_dataset():
    try:
        print('Downloading data from a2i...')
        df = pd.read_csv('http://cdr.a2i.gov.bd/positive_case_data/'+datadate+'.csv').dropna()

        replace_district_name = {'Kishoreganj': 'KISHOREGANJ'}
        df = df.replace({'district': replace_district_name})

        df.to_csv('data/'+datadate+'.csv', index=False)

    except:
        ss.exit(datadate+".csv does not exist in the server. a 404 error")

    try:
        print('Downloading data from owid...')
        df_owid = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
        df_owid = df_owid[df_owid.location=='Bangladesh']
        df_owid.to_csv('data/'+datadate+'_owid.csv', index=False)
    except:
        ss.exit('owid does not exist...')
    return df, df_owid

if(path.exists('data/'+datadate+'.csv')):
    print('Fetching dataset...')
    df_a2i = pd.read_csv('data/'+datadate+'.csv').dropna()
    df_owid = pd.read_csv('data/'+datadate+'_owid.csv')
else:
    df_a2i, df_owid = download_a2i_dataset()

districts = df_a2i.district.sort_values().unique()
datadate = df_a2i.test_date.iloc[-1]

try:
    ### Districs population dataset
    df_population = pd.read_csv('data/district_population.csv').dropna()
    df_population = df_population[['Name', 'Population\nCensus (Cf)\n2011-03-15', 'Abbr.']]
    df_population['population'] = pd.to_numeric(df_population['Population\nCensus (Cf)\n2011-03-15'].str.replace(',', ''))

    replace_district_name = {
            'Jhalakati (Jhalokati)': 'Jhalokati',
            'Cox\'s Bazar': 'Coxs bazar',
            'Netrakona (Netrokona)': 'Netrakona',
            'Jhenaidah (Jhenida)': 'Jhenaidah',
            'Chapai Nawabganj': 'Chapainababganj',
            'Jaipurhat (Joypurhat)': 'Joypurhat',
            'Barisal': 'Barishal',
            'Bogra': 'Bogura',
            'Chittagong': 'Chattogram',
            'Comilla': 'Cumilla',
            'Jessore': 'Jashore',
            'Maulvi Bazar (Moulvibazar)': 'Moulvibazar'
            }

    df_population = df_population.replace({'Name': replace_district_name})
except:
    ss.exit("district_population.csv does not exist in the server. Check the file.")



### obtain the TPR for last 14 days
def test_positive_rate():

    df_case = df_a2i

    latest_data_date = df_case.test_date.sort_values().unique()[-1]
    last_data_date = pd.to_datetime(latest_data_date) - timedelta(days=14)
    last_data_date = last_data_date.strftime('%Y-%m-%d')

    # print(latest_data_date, last_data_date)

    df = df_case[df_case.test_date > last_data_date]

    df['TPR'] = df.cases_nontraveller/df.tests_nontraveller
    df['district'] = df['district'].str.capitalize()

    # print(df.district.sort_values().unique(), df[df.district == 'Khagrachhari'])

    return df


def districts_tpr():
    # df = pd.read_csv('2021-04-02.csv').dropna()
    df = test_positive_rate()

    # df['TPR_cases'] = df.cases_nontraveller*df.cases_nontraveller/df.tests_nontraveller
    df['TPR_nontraveller'] = df.TPR * df.cases_nontraveller
    df['TPR_combined'] = df.cases_combined/df.tests_combined
    df['TPR_combined_cases'] = df.TPR_combined * df.cases_combined
    # df['TPR'] = df.cases_nontraveller/df.tests_nontraveller

    # print(df)

    def data_plot(x, y, color, label, ax=plt):
        mask = ~np.isnan(x) & ~np.isnan(y)
        stats = linregress(x[mask], y[mask])
        print(stats)

        m = stats.slope
        b = stats.intercept
        # err = stats.stderr/2
        r2 = stats.rvalue*stats.rvalue
        p = stats.pvalue
        label = 'm: {:.2}, $R^2$: {:.2}'.format(m, r2)
        # label = '$R^2$: {:.2}'.format(r2)


        # ax.scatter(x, y, marker='.', color=color, label=None)
        l, = ax.plot(x, m * x + b, color=color, label=label)
        # ax.fill_between(x, m*x+b+err, m*x+b-err, alpha=0.2, color=color)
        ax.set_title(label)
        return m


    df['district'] = pd.Categorical(df['district'])
    df['date'] = pd.to_datetime(df.test_date)

    district = np.array(df['district'])

    # print(df.district)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    sns.scatterplot(x='cases_nontraveller', y='TPR', hue=district, data=df, legend=False, ax=axs[0, 0])
    data_plot(df.cases_nontraveller, df.TPR, 'r', False, axs[0, 0])

    sns.scatterplot(x='cases_nontraveller', y='TPR_nontraveller', hue=district, data=df, legend=False, ax=axs[0, 1])
    m = data_plot(df.cases_nontraveller, df.TPR_nontraveller, 'r', False, axs[0, 1])

    sns.scatterplot(x='cases_combined', y='TPR_combined', hue=district, data=df, legend=False, ax=axs[1, 0])
    data_plot(df.cases_combined, df.TPR_combined, 'r', False, axs[1, 0])

    sns.scatterplot(x='cases_combined', y='TPR_combined_cases', hue=district, data=df, legend=False, ax=axs[1, 1])
    data_plot(df.cases_combined, df.TPR_combined_cases, 'r', False, axs[1, 1])

    # plt.show()
    return m

# districts_tpr()

def test_k_means(x, n_clusters=4, ax=plt):
    from sklearn.cluster import KMeans

    x = x.dropna().to_numpy()

    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, algorithm='full')
    a = kmeans.fit(np.reshape(x,(len(x),1)))
    centroids = kmeans.cluster_centers_

    labels = kmeans.labels_

    # print(centroids)
    # print(labels)

    import matplotlib.colors as mcolors
    colors = mcolors.TABLEAU_COLORS
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name) for name, color in colors.items())
    names = [name for hsv, name in by_hsv]

    # print(names)
    colors = names[-len(centroids):]
    # colors = ["g","y","orange","r"]

    # for i in centroids: ax.plot( [0, len(x)-1],[i,i], "k" )
    # for i in range(len(x)):
    #     # print(i, x[i])
    #     ax.semilogy(i, x[i], colors[labels[i]], marker='.', markersize = 10)

    # ax.set_xlabel('Number of Data')
    # ax.set_ylabel('Value')

    df = pd.DataFrame({'x': x, 'labels': labels})
    df = df.sort_values('x')
    unique_labels = df.labels.unique()
    bins = []
    for i in range(len(unique_labels)-1):
        # print(unique_label, df[df.labels == unique_label]['x'].max())
        bins.append(df[df.labels == unique_labels[i]]['x'].max())

    print(bins)

    return bins


# test_k_means()


def prepare_weekly_df(df_case, first_data_date='2020-04-01', days=6):
    last_data_date = pd.to_datetime(first_data_date) + timedelta(days=days)
    last_data_date = last_data_date.strftime('%Y-%m-%d')

    # print(df_case.test_date.sort_values().unique())
    # print(first_data_date, last_data_date, df_case[(df_case.test_date > first_data_date) & (df_case.test_date <= last_data_date)].test_date.unique())

    return df_case[(df_case.test_date >= first_data_date) & (df_case.test_date <= last_data_date)], last_data_date

# test()

def data_plot(x, y, color, label, ax=plt):
    mask = ~np.isnan(x) & ~np.isnan(y)
    stats = linregress(x[mask], y[mask])
    print(stats)

    m = stats.slope
    b = stats.intercept
    # err = stats.stderr/2
    r2 = stats.rvalue*stats.rvalue
    p = stats.pvalue
    # title = 'm: {:.2}, $R^2$: {:.2}'.format(m, r2)
    # label = '$R^2$: {:.2}'.format(r2)

    # ax.scatter(x, y, marker='.', color=color, label=None)
    # l, = ax.plot(x, m * x + b, color=color, label=label)
    # # ax.fill_between(x, m*x+b+err, m*x+b-err, alpha=0.2, color=color)
    # ax.title(title)
    return m

def TPR_nontraveller():
    tpr = data_plot(df_a2i.tests_nontraveller, df_a2i.cases_nontraveller, 'r', None)

    return tpr

# TPR_nontraveller()

def TPR_combined():
    tpr = data_plot(df_a2i.tests_combined, df_a2i.cases_combined, 'b', None)

    return tpr

# TPR_combined()


