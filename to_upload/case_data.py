import pandas as pd

def data_length(df, date):
    count = 0
    for index in df.columns[4:]:
        if pd.to_datetime(index) < pd.to_datetime(date):
            count += 1

    return df[df.columns[:3+count]]

def data_fetch(date):
    CONFIRMED_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    CONFIRMED_URL_DATA = pd.read_csv(CONFIRMED_URL)

    RECOVERED_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    RECOVERED_URL_DATA = pd.read_csv(RECOVERED_URL)

    DEATHS_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    DEATHS_URL_DATA = pd.read_csv(DEATHS_URL)


    data_length(CONFIRMED_URL_DATA, date).to_csv('data/confirmed.csv', index=False)
    data_length(RECOVERED_URL_DATA, date).to_csv('data/recovered.csv', index=False)
    data_length(DEATHS_URL_DATA, date).to_csv('data/deaths.csv', index=False)

data_fetch('6/1/2020')
