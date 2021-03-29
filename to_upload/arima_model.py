import pandas as pd
from pandas.plotting import lag_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
import datetime
from datetime import date

def load_series():
    # Load series to predict
    series = pd.read_csv('data/Bangladesh_90_1.csv')
    confirmed = series['confirmed'].values
    dates = series['days'].values
    return confirmed, dates

def SARIMAX_model(series, order, days = 30):
    # Fitting and forecast the series
    train = [x for x in series]
    model = SARIMAX(train, order = order)
    model_fit = model.fit(disp=0)
    forecast = model_fit.forecast(steps = days, alpha = 0.05)
    # print(forecast)
    # start_day = date.today() + datetime.timedelta(days = 1)
    start_day = pd.to_datetime(dates_today) + datetime.timedelta(days = 1)
    predictions_df = pd.DataFrame({'Forecast':forecast.round()}, index=pd.date_range(start = start_day, periods=days, freq='D'))
    return predictions_df

def ARIMA_model(series, order, days = 30):
    # Fitting and forecast the series
    train = [x for x in series]
    model = ARIMA(train, order = order)
    model_fit = model.fit(disp=0)
    forecast, err, ci = model_fit.forecast(steps = days, alpha = 0.05)
    # print(forecast)
    # start_day = date.today() + datetime.timedelta(days = 1)
    start_day = pd.to_datetime(dates_today) + datetime.timedelta(days = 1)
    predictions_df = pd.DataFrame({'Forecast':forecast.round()}, index=pd.date_range(start = start_day, periods=days, freq='D'))
    return predictions_df, ci

def plot_results(series, df_forecast, ci, label, filename):
    # start_covid_day = date(2020, 2, 24)
    start_covid_day = pd.to_datetime(dates[0])
    series = pd.DataFrame({'Real data':series}, index=pd.date_range(start = start_covid_day, periods=series.shape[0], freq='D'))
    # print(series)
    df_forecast.to_csv('data/'+filename+'.csv')
    ax = series.plot(label = 'Real Data')
    df_forecast.plot(ax = ax, label='Forecast', color = 'r')

    if (ci is None):
        pass
    else:
        ax.fill_between(df_forecast.index,
                        ci[:,0],
                        ci[:,1], color='b', alpha=.25)
    ax.set_xlabel('Days')
    ax.set_ylabel(label)
    ax.set_title(label + ' Forecasting')
    plt.legend()
    # plt.savefig('plots/' + label + '.png')
    plt.show()

# Order for ARIMA model
order = {
    'new_positives': (3, 1, 0)
}

# Dowload and loading series
confirmed, dates = load_series()

# Stats of today
confirmed_today, dates_today = confirmed[-1], dates[-1]

# Forecasting with ARIMA models
confirmed_pred, confirmed_ci = ARIMA_model(confirmed, order['new_positives'])
confirmed_pred = SARIMAX_model(confirmed, order['new_positives'])

# Plot Results
plot_results(confirmed, confirmed_pred, confirmed_ci, 'Confirmed', 'arima')
plot_results(confirmed, confirmed_pred, None, 'Confirmed', 'sarimax')
