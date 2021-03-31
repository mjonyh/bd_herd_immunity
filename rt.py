from scipy import stats as sps
from scipy.interpolate import interp1d
import logging

import warnings
import pandas as pd
import numpy as np
import altair as alt
import sys

warnings.simplefilter(action='ignore', category=FutureWarning)

# dataset = pd.read_csv('datasetcv.csv')
dataset = pd.read_csv('data/Bangladesh_90_'+sys.argv[1]+'.csv')
if(sys.argv[1]=='1'):
    dataset['Date'] = pd.to_datetime(dataset['days']).dt.strftime('%Y-%m-%d')
    dataset['confirmed'] = pd.to_numeric(dataset['confirmed'])
    dataSeries = pd.Series(dataset['confirmed'].values, index=dataset['Date'])
else:
    dataset['Date'] = pd.to_datetime(dataset['days_sim']).dt.strftime('%Y-%m-%d')
    dataset['confirmed_sim'] = pd.to_numeric(dataset['confirmed_sim'])
    dataSeries = pd.Series(dataset['confirmed_sim'].values, index=dataset['Date'])
# print(dataSeries)


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

# dataSeries = pd.Series(datasetxl[district_name].values, index=datasetxl['Date'])

original, smoothed = prepare_cases(dataSeries)

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
result.to_csv('data/rt_'+sys.argv[1]+'.csv')
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
df.to_csv('data/doublingtimes_'+sys.argv[1]+'.csv')

# plt.show()

