import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df_age = pd.read_csv('data/Bangladesh_age_distribution.csv')


df_age['Percentage'] = df_age['Population']*100/df_age['Population'].sum()

# df_age['AgeGroup'] = ['1 - 20', '1 - 20', '1 - 20', '1 - 20', '21 - 40', '21 - 40', '21 - 40', '21 - 40', '41 - 60', '41 - 60', '41 - 60', '41 - 60', '60+', '60+', '60+', '60+', '60+']
df_age['AgeGroup'] = ['1 - 10', '1 - 10', '11 - 20', '11 - 20', '21 - 30', '21 - 30', '31 - 40', '31 - 40', '41 - 50', '41 - 50', '51 - 60', '51 - 60', '60+', '60+', '60+', '60+', '60+']
df = df_age.groupby('AgeGroup').sum()

df['Cases'] = [ 2.9, 7.3, 27.6, 27.1, 17.3, 11.1, 6.7 ]
df['Deaths'] =[ 1.01, 1.85, 3.52, 8.05, 17.11, 31.38, 37.08 ]
# df['Cases'] = [ 10.2, 54.6, 28.4, 6.7 ]
# df['Deaths'] =[ 2.86, 11.57, 48.49, 37.08 ]


### values
divisor = 1e6
percent_f = 29.0

# predicted distribution of population in 2020
weight = 170.1/153.0
df['Pop_2020'] = df['Population'] * weight
df['Pop_2020_m'] = df['Pop_2020'] / 2.0
df['Pop_2020_f'] = df['Pop_2020'] / 2.0
print(df[['Population', 'Pop_2020', 'Pop_2020_m']])

# prediction of infected diseases at age 21-30
m_21_30_infected = df['Pop_2020_m'][2] * 90 / 100.0
f_21_30_infected = m_21_30_infected * percent_f / 100.0

df['Sus_m'] = df['Cases'] * m_21_30_infected / (df['Cases'][2])
df['Sus_f'] = df['Cases'] * f_21_30_infected / (df['Cases'][2])
df['Sus'] = df['Sus_m'] + df['Sus_f']

df['Sus_percentage'] = df['Sus']*100.0/df['Pop_2020']
df['Sus_m_percentage'] = df['Sus_m']*100.0/df['Pop_2020_m']
df['Sus_f_percentage'] = df['Sus_f']*100.0/df['Pop_2020_f']

df_1 = df/1e6

df_1.plot.bar(y=['Pop_2020', 'Pop_2020_m', 'Pop_2020_f', 'Sus', 'Sus_m', 'Sus_f'], label=['Population', 'Male', 'Female', 'Susceptible Population', 'Susceptible Male', 'Susceptible Female'])

plt.xticks(rotation=45)
plt.xlabel('Age Group')
plt.ylabel('Population (in million)')

df.plot.bar(y=['Percentage', 'Cases', 'Sus_percentage', 'Sus_m_percentage', 'Sus_f_percentage'], label=['Population', 'Confirmed Cases', 'Susceptible Population', 'Susceptible Male', 'Susceptible Female'])

plt.xticks(rotation=45)
plt.xlabel('Age Group')
plt.ylabel('Percentage (%)')

print(df['Percentage'])

for i in range(30, 100, 10):
    percent_m = i
    # m_21_30_infected = df['Pop_2020_m'][2] * percent_m / 100.0
    # f_21_30_infected = m_21_30_infected * percent_f / 100.0
    # t_21_30_infected = m_21_30_infected + f_21_30_infected

    # df['Pop'] = df['Population'] / divisor
    # df['Sus'] = df['Cases'] * t_21_30_infected / (df['Cases'][2] * divisor)

    m_21_30_infected = df['Pop_2020_m'][2] * i / 100.0
    f_21_30_infected = m_21_30_infected * percent_f / 100.0

    df['Sus_m'] = df['Cases'] * m_21_30_infected / (df['Cases'][2])
    df['Sus_f'] = df['Cases'] * f_21_30_infected / (df['Cases'][2])
    df['Sus'] = df['Sus_m'] + df['Sus_f']
    # df.plot.bar(y=['Pop', 'Sus'])

    # plt.xticks(rotation=45)
    # plt.xlabel('Age Group')
    # plt.ylabel('Frequency (x1 million)')
    # df.plot.bar(y='Susceptible')
    # print(i, df['Sus'].sum()/divisor)
    print(i, df['Sus'])
plt.show()
