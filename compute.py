from sird_model import *
import sys

# Create an instance of the SIRD model, asking for the data to be used.
# plt.style.use('grayscale')

# countries = ['Bangladesh']
country = 'Bangladesh'
# run_time = [300, 140, 115]
run_time = 730
# population = 170.1e6
### Bangladesh susceptible population 5,26,73,000

age_range = [ '1 - 10', '11 - 20', '21 - 30', '31 - 40', '41 - 50', '51 - 60', '60+' ]
age_case_distribution = np.array([ 2.9, 7.3, 27.6, 27.1, 17.3, 11.1, 6.7 ])
age_death_distribution = np.array([ 1.01, 1.85, 3.52, 8.05, 17.11, 31.38, 37.08 ])

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# colors = ['tab:red', 'tab:blue']
# plt.setp(ax1.xaxis.get_majorticklabels(),rotation=45,horizontalalignment='right')
# markers = ['-', '-.', '--']
# tag = 0
# save_percent = ['30', '80']
# people = [17.7e6, 53.0e6]
# for country in countries:
# for i in range(len(save_percent)):
# save_percent = '30'
save_percent = sys.argv[1]

if(save_percent=='30'):
    people = 17.7e6
else:
    people = 53.0e6

m = Model(country=country, people=people, tag=0)

m.reset()

# Run our SIRD model and plot its S, I, R and D values, together with the data.

m.run(nb_of_days=run_time)
# m.plot()
# print(len(m.beta()), len(m.days_array()))
# print(m.active_data())
index = len(m.confirmed_data())
print('R0 = ', m.beta(index)/(m.gamma(index) + m.mu(index)))
print('beta: ', m.beta(index))
print('gamma: ', m.gamma(index))
print('mu: ', m.mu(index))

days_array = m.days_array()
days_case = m.days_cases()


data_1 = {
        'days': days_case,
        'confirmed': m.confirmed_data(),
        'recovered': m.recovered_data(),
        'deaths': m.deaths_data(),
        'active': m.active_data(),
        'cfr_recovered': m.deaths_data()*100/(m.recovered_data()+m.deaths_data()), 
        'cfr_confirmed': m.deaths_data()*100/m.confirmed_data()
        }

df = pd.DataFrame(data_1)
df.to_csv('data/'+country+'_'+save_percent+'_1.csv', index=False)

data_2 = {
        'days_sim': days_array,
        'susceptible_sim': m.s(),
        'active_sim': m.i(),
        'recovered_sim': m.r(),
        'deaths_sim': m.d(),
        'confirmed_sim': m.i()+m.r()+m.d(),
        'beta': m.beta(),
        'gamma': m.gamma(),
        'mu': m.mu(),
        'r0': m.beta()/(m.gamma() + m.mu()),
        're': (m.beta()/(m.gamma()+m.mu()))*(m.s()/m.s(0)),
        'cfr_recovered_sim': m.d()*100/(m.r()+m.d()),
        'cfr_confirmed_sim': m.d()*100/(m.r()+m.i()+m.d())
        }

df = pd.DataFrame(data_2)
df.to_csv('data/'+country+'_'+save_percent+'_2.csv', index=False)

plt.show()

