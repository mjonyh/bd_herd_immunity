from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy import stats as sps
from scipy.interpolate import interp1d
import logging

# from IPython.display import clear_output
import matplotlib.font_manager as fm
import warnings
import pandas as pd
import numpy as np
import altair as alt

from matplotlib.ticker import MaxNLocator
warnings.simplefilter(action='ignore', category=FutureWarning)

# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline

# dataset = pd.read_csv('datasetcv.csv')
dataset = pd.read_csv('data/Bangladesh_80_1.csv')
dataset['Date'] = pd.to_datetime(dataset['days']).dt.strftime('%Y-%m-%d')
dataset['confirmed'] = pd.to_numeric(dataset['confirmed'])
# print(dataset.head())
dataSeries = pd.Series(dataset['confirmed'].values, index=dataset['Date'])
# print(dataSeries)

# datasetxl = pd.read_excel('dataset.xlsx','Sheet1')
# datasetxl = datasetxl.dropna(how='all', axis='columns')
# datasetxl = datasetxl.loc[:, ~datasetxl.columns.str.contains('^Unnamed',na=False)]
# datasetxl = datasetxl.drop([0,67,68,69])
# datasetxl.to_csv("dataset.csv")
# datasetxl = pd.read_csv("dataset.csv",header=None)
# datasetxl = datasetxl.transpose()
# datasetxl = datasetxl.drop([0,1])
# datasetxl = datasetxl.reindex(index=datasetxl.index[::-1])
# datasetxl.columns=["Date","B. Baria","Bagerhat","Bandarban","Barguna","Barisal","Bhola","Bogra","Chandpur","Chapainawabganj","Chattogram","Chuadanga","Cox’s bazar","Cumilla","Dhaka (District)","Dhaka City","Dinajpur","Faridpur","Feni","Gaibandha","Gazipur","Gopalganj","Habiganj","Jamalpur","Jessore","Jhalokathi","Jhenaidah","Joypurhat","Khagrachhari","Khulna","Kishoreganj","Kurigram","Kushtia","Lakshmipur","Lalmonirhat","Madaripur","Magura","Manikganj","Meherpur","Moulvibazar","Munshiganj","Mymensingh","Naogaon","Narail","Narayanganj","Narsingdi","Natore","Netrokona","Nilphamari","Noakhali","Pabna","Panchagarh","Pirojpur","Potuakhali","Rajbari","Rajshahi","Rangamati","Rangpur","Satkhira","Shariatpur","Sherpur","Sirajganj","Sunamganj","Sylhet","Tangail","Thakurgaon","total"]
# datasetxl['Date'] = pd.to_datetime(datasetxl['Date']).dt.strftime('%Y-%m-%d')
# districts = ['B. Baria','Bagerhat','Bandarban','Barguna','Barisal','Bhola','Bogra','Chandpur','Chapainawabganj','Chattogram','Chuadanga','Cox’s bazar','Cumilla','Dhaka (District)','Dhaka City','Dinajpur','Faridpur','Feni','Gaibandha','Gazipur','Gopalganj','Habiganj','Jamalpur','Jessore','Jhalokathi','Jhenaidah','Joypurhat','Khagrachhari','Khulna','Kishoreganj','Kurigram','Kushtia','Lakshmipur','Lalmonirhat','Madaripur','Magura','Manikganj','Meherpur','Moulvibazar','Munshiganj','Mymensingh','Naogaon','Narail','Narayanganj','Narsingdi','Natore','Netrokona','Nilphamari','Noakhali','Pabna','Panchagarh','Pirojpur','Potuakhali','Rajbari','Rajshahi','Rangamati','Rangpur','Satkhira','Shariatpur','Sherpur','Sirajganj','Sunamganj','Sylhet','Tangail','Thakurgaon','total']
# datasetxl[districts] = datasetxl[districts].fillna(0.0)
# datasetxl[districts] = datasetxl[districts].apply(pd.to_numeric, errors='coerce')
# datasetxl[districts] = datasetxl[districts].cumsum()

#hide_input
def prepare_cases(cases, cutoff=5):
    new_cases = cases.diff()

    smoothed = new_cases.rolling(7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()
    
    idx_start = np.searchsorted(smoothed, cutoff)
    
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    
    return original, smoothed

# district_name = 'Dhaka City'
# district_name = 'Narayanganj'

# dataSeries = pd.Series(datasetxl[district_name].values, index=datasetxl['Date'])

original, smoothed = prepare_cases(dataSeries)

#for district
# original.plot(title=district_name + " New Cases per Day",
original.plot(title=" New Cases per Day",
               c='k',
               linestyle=':',
               alpha=.5,
               label='Actual',
               legend=True,
             figsize=(500/72, 300/72))

ax = smoothed.plot(label='Smoothed',
                   legend=True)

ax.get_figure().set_facecolor('w')

#hide_input

# We create an array for every possible value of Rt
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316
GAMMA = 1/7

#hide_input
def get_posteriors(sr, sigma=0.15):

    # (1) Calculate Lambda
    # Map Rt into lambda so we can substitute it into the equation below
    # Note that we have N-1 lambdas because on the first day of an outbreak
    # you do not know what to expect.
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

  
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])
  
    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
  
    # (4) Calculate the initial prior
    #prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 = np.ones_like(r_t_range)/len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )
  
    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = np.dot(process_matrix, posteriors[previous_day])
      
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
      
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
      
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator
      
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
        # print(log_likelihood)
  
    return posteriors, log_likelihood

# Note that we're fixing sigma to a value just for the example
posteriors, log_likelihood = get_posteriors(smoothed, sigma=.25)

# ax = posteriors.plot(title=district_name + ' - Daily Posterior for $R_t$',
ax = posteriors.plot(title=' - Daily Posterior for $R_t$',
           legend=False, 
           lw=1,
           c='k',
           alpha=.3,
           xlim=(0.4,6))

ax.set_xlabel('$R_t$');

# MLE to find sigma
sigmas = np.linspace(1/20, 1, 20)

new, smoothed = prepare_cases(dataSeries, cutoff=7)
if len(smoothed) == 0:
    new, smoothed = prepare_cases(dataSeries, cutoff=10)

result = {}

# Holds all posteriors with every given value of sigma
result['posteriors'] = []

# Holds the log likelihood across all k for each value of sigma
result['log_likelihoods'] = []

for sigma in sigmas:
    posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)
    result['posteriors'].append(posteriors)
    result['log_likelihoods'].append(log_likelihood)

# clear_output(wait=True)
sigma = sigmas[np.argmax(result['log_likelihoods'])]
posteriors = result['posteriors'][np.argmax(result['log_likelihoods'])]
logging.debug("Sigma: {sigma} has highest log likelihood")
logging.debug('Done.')

#hide_input
# Calculate High density interval
def highest_density_interval(pmf, p=.9, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
  
    cumsum = np.cumsum(pmf.values)
  
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]
  
    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
  
    # Find the smallest range (highest density)
    best = (highs - lows).argmin()
  
    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]
  
    return pd.Series([low, high],
                     index=['Low_90',
                            'High_90'])



# Note that this takes a while to execute - it's not the most efficient algorithm
hdis = highest_density_interval(posteriors, p=.9)

most_likely = posteriors.idxmax().rename('ML')

# Look into why you shift -1
result = pd.concat([most_likely, hdis], axis=1)

# print(result)

def plot_rt(result, ax, district_name):
  
    ax.set_title(str(district_name))
  
    result.to_csv('data/rt.csv')
    # Colors
    ABOVE = [1,0,0]
    MIDDLE = [1,1,1]
    BELOW = [0,0,0]
    cmap = ListedColormap(np.r_[
        np.linspace(BELOW,MIDDLE,25),
        np.linspace(MIDDLE,ABOVE,25)
    ])
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
  
    index = result['ML'].index.get_level_values('Date')
    # print(index, type(index))
    # index = index.to_datetime()
    index = pd.to_datetime(index)
    values = result['ML'].values
    #print(index)
  
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)
    #"""
    # Aesthetically, extrapolate credible interval by 1 day either side
    # lowfn = interp1d(date2num(index.to_datetime()),
    lowfn = interp1d(date2num(pd.to_datetime(index)),
                     result['Low_90'].values,
                     bounds_error=False,
                     fill_value='extrapolate')
  
    highfn = interp1d(date2num(pd.to_datetime(index)),
                      result['High_90'].values,
                      bounds_error=False,
                      fill_value='extrapolate')
  
    extended = pd.date_range(start=pd.Timestamp('2020-03-01'),
                             end=index[-1]+pd.Timedelta(days=1))
  
    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);
  
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
  
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlim(pd.Timestamp('2020-03-01'), pd.Timestamp(result.index.get_level_values('Date')[-1])+pd.Timedelta(days=1))
    #"""
    fig.set_facecolor('w')
    fig.savefig('bangladesh-rt.png')  

fig, ax = plt.subplots(figsize=(600/72,400/72))
#result.index = pd.to_datetime(result.index)

# plot_rt(result, ax, district_name)
plot_rt(result, ax, 'Bangladesh')
_ = plt.xticks(rotation=45, ha='right')
#_ = plt.xticks(np.arange(0, len(dates), step=3))
# ax.set_title('Real-time $R_t$ for ' + district_name)
ax.set_title('Real-time $R_t$ for ')
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))


### Doubling
print(original.index.to_numpy())

# date = original.index.get_values()
# cases = original.get_values()
date = original.index.to_numpy()
cases = original.to_numpy()

allcases = [[date[0], cases[0], 0, 0]]
for i in range(1, len(date)):
    # print(allcases[-1])
    allcases.append([date[i], allcases[-1][1] + cases[i], 0, 0])

for i in range(len(allcases)):
    if allcases[i][2] == 0.0:
        allcases[i][2] = allcases[i][1]/allcases[i-1][1]-1
    if allcases[i][3] == 0.0:
        allcases[i][3] = 0.7/allcases[i][2];

doublingtimes = [row[3] for row in allcases]
dates = [row[0] for row in allcases];

# print(type(doublingtimes), type(dates))

data = {
        'date': dates,
        'doublingtimes': doublingtimes
        }

df = pd.DataFrame(data)
df.to_csv('data/doublingtimes.csv')

plt.figure(figsize=(6, 4))
plt.autoscale(enable=True, axis='x', tight=True)
plt.rcParams.update({'font.size': 12})
plt.plot(dates, doublingtimes)#, '-', color="#348ABD", label='$Herd Immunity$', lw=4)
plt.xlabel('Date Range')
plt.ylabel('Doubling Time for ')
_ = plt.xticks(rotation=45, ha='right')
_ = plt.xticks(np.arange(0, len(dates), step=3))




plt.show()

