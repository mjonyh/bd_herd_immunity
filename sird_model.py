import math
import sys
from enum import Enum, auto

import matplotlib
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def convert_to_date(starting_date, date_length):
    date_array = []
    for i in range(len(date_length)):
        date_array.append(starting_date + np.timedelta64(i, 'D'))

    return date_array

class Model:
    """
    SIRD model of Covid-19.
    """

    __CONFIRMED_URL = 'https://bit.ly/35yJO0d'
    __CONFIRMED_URL_DATA = pd.read_csv(__CONFIRMED_URL)
    __RECOVERED_URL = 'https://bit.ly/2L6jLE9'
    __RECOVERED_URL_DATA = pd.read_csv(__RECOVERED_URL)
    __DEATHS_URL = 'https://bit.ly/2L0hzxQ'
    __DEATHS_URL_DATA = pd.read_csv(__DEATHS_URL)
    __POPULATION_URL = 'https://bit.ly/2WYjZCD'
    __JHU_DATA_SHIFT = 4
    __N_FILTERED = 7  # Number of state variables to filter (I, R, D, β, γ, μ and n, the population).
    __N_MEASURED = 3  # Number of measured variables (I, R and D).
    __NB_OF_STEPS = 100
    __DELTA_T = 1 / __NB_OF_STEPS
    __FIG_SIZE = (11, 13)
    __S_COLOR = '#0072bd'
    __I_COLOR = '#d95319'
    __R_COLOR = '#edb120'
    __D_COLOR = '#7e2f8e'
    __BETA_COLOR = '#77ac30'
    __GAMMA_COLOR = '#4dbeee'
    __MU_COLOR = '#a2142f'
    __DATA_ALPHA = 0.3
    __DATA = None
    __POPULATION = None

    class Use(Enum):
        WIKIPEDIA = auto()
        DATA = auto()

    def __init__(self, use=Use.DATA, country='Bangladesh', max_data=0, people=1e6, tag=1):
        """
        Initialise our Model object.
        """

        # Retrieve the data (if requested and needed).

        if use == Model.Use.DATA and Model.__DATA is None:
            confirmed_data, confirmed_data_start = self.__jhu_data(Model.__CONFIRMED_URL_DATA, country)
            recovered_data, recovered_data_start = self.__jhu_data(Model.__RECOVERED_URL_DATA, country)
            deaths_data, deaths_data_start = self.__jhu_data(Model.__DEATHS_URL_DATA, country)
            data_start = min(confirmed_data_start, recovered_data_start, deaths_data_start) - Model.__JHU_DATA_SHIFT
            start_date = confirmed_data.columns[data_start].split('/')

            for i in range(data_start, confirmed_data.shape[1]):
                c = confirmed_data.iloc[0][i]
                r = recovered_data.iloc[0][i]
                d = deaths_data.iloc[0][i]
                data = [c - r - d, r, d]
                # print(c, r, d)

                if Model.__DATA is None:
                    Model.__DATA = np.array(data)
                else:
                    Model.__DATA = np.vstack((Model.__DATA, data))

        # print(Model.__DATA)
        # Model.__DATA = smooth_data(Model.__DATA)
        if use == Model.Use.DATA:
            self.__data = Model.__DATA
        else:
            self.__data = None

        if self.__data is not None:
            if not isinstance(max_data, int):
                sys.exit('Error: \'max_data\' must be an integer value.')

            if max_data > 0:
                self.__data = self.__data[:max_data]

        # Retrieve the population (if needed).

        if (tag == 0):

            if use == Model.Use.DATA:
                Model.__POPULATION = {}
                # print(tag)
                tag = 1

                # response = requests.get(Model.__POPULATION_URL)
                # soup = BeautifulSoup(response.text, 'html.parser')
                # data = soup.select('div div div div div tbody tr')

                # for i in range(len(data)):
                #     country_soup = BeautifulSoup(data[i].prettify(), 'html.parser')
                #     country_value = country_soup.select('tr td a')[0].get_text().strip()
                #     population_value = country_soup.select('tr td')[2].get_text().strip().replace(',', '')

                #     # Model.__POPULATION[country_value] = int(population_value)
                #     # print(people)
                #     # Model.__POPULATION[country_value] = int(people)
                Model.__POPULATION[country] = int(people)


            if use == Model.Use.DATA:
                if country in Model.__POPULATION:
                    self.__population = Model.__POPULATION[country]
                else:
                    sys.exit('Error: no population data is available for {}.'.format(country))

        # Keep track of whether to use the data.

        self.__use_data = use == Model.Use.DATA

        # Declare some internal variables (that will then be initialised through our call to reset()).

        self.__beta = None
        self.__gamma = None
        self.__mu = None

        self.__ukf = None

        self.__x = None
        self.__n = None

        self.__data_s_values = None
        self.__data_i_values = None
        self.__data_r_values = None
        self.__data_d_values = None

        self.__s_values = None
        self.__i_values = None
        self.__r_values = None
        self.__d_values = None

        self.__beta_values = None
        self.__gamma_values = None
        self.__mu_values = None

        # Initialise (i.e. reset) our SIRD model.

        self.__start_date = pd.to_datetime(start_date[0]+'-'+start_date[1]+'-'+start_date[2])
        self.__confirmed_data = confirmed_data.iloc[0][data_start:-1]
        self.__recovered_data = recovered_data.iloc[0][data_start:-1]
        self.__deaths_data = deaths_data.iloc[0][data_start:-1]

        self.reset()

    @staticmethod
    def __jhu_data(data, country):
        # data = pd.read_csv(url)
        # data = data[(data['Country/Region'] == country) & data['Province/State'].isnull()]
        data_1 = data[(data['Country/Region'] == country) & data['Province/State'].isnull()]
        if data_1.shape[0] == 0:
            data = data[(data['Country/Region'] == country)].groupby('Country/Region').sum()
            data.to_csv('test_data.csv')
            data = pd.read_csv('test_data.csv')
        else:
            data = data_1

        if data.shape[0] == 0:
            sys.exit('Error: no Covid-19 data is available for {}.'.format(country))

        data = data.drop(data.columns[list(range(Model.__JHU_DATA_SHIFT))], axis=1)  # Skip non-data columns.
        start = None

        for i in range(data.shape[1]):
            if data.iloc[0][i] != 0:
                start = Model.__JHU_DATA_SHIFT + i

                break

        return data, start

    def __data_x(self, day, index):
        """
        Return the I/R/D value for the given day.
        """

        return self.__data[day][index] if self.__use_data else math.nan

    def __data_s(self, day):
        """
        Return the S value for the given day.
        """

        if self.__use_data:
            return self.__population - self.__data_i(day) - self.__data_r(day) - self.__data_d(day)
        else:
            return math.nan

    def __data_i(self, day):
        """
        Return the I value for the given day.
        """

        return self.__data_x(day, 0)

    def __data_r(self, day):
        """
        Return the R value for the given day.
        """

        return self.__data_x(day, 1)

    def __data_d(self, day):
        """
        Return the D value for the given day.
        """

        return self.__data_x(day, 2)

    def __data_available(self, day):
        """
        Return whether some data is available for the given day.
        """

        return day <= self.__data.shape[0] - 1 if self.__use_data else False

    def __s_value(self):
        """
        Return the S value based on the values of I, R, D and N.
        """

        return self.__n - self.__x.sum()

    def __i_value(self):
        """
        Return the I value.
        """

        return self.__x[0]

    def __r_value(self):
        """
        Return the R value.
        """

        return self.__x[1]

    def __d_value(self):
        """
        Return the D value.
        """

        return self.__x[2]

    @staticmethod
    def __f(x, dt, **kwargs):
        """
        State function.

        The ODE system to solve is:
          dI/dt = βIS/N - γI - μI
          dR/dt = γI
          dD/dt = μI
        """

        model_self = kwargs.get('model_self')
        with_ukf = kwargs.get('with_ukf', True)

        if with_ukf:
            s = x[6] - x[:3].sum()
            beta = x[3]
            gamma = x[4]
            mu = x[5]
            n = x[6]
        else:
            s = model_self.__n - x.sum()
            beta = model_self.__beta
            gamma = model_self.__gamma
            mu = model_self.__mu
            n = model_self.__n

        a = np.array([[1 + dt * (beta * s / n - gamma - mu), 0, 0, 0, 0, 0, 0],
                      [dt * gamma, 1, 0, 0, 0, 0, 0],
                      [dt * mu, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1]])

        if with_ukf:
            return a @ x
        else:
            return a[:3, :3] @ x

    @staticmethod
    def __h(x):
        """
        Measurement function.
        """

        return x[:Model.__N_MEASURED]

    def reset(self):
        """
        Reset our SIRD model.
        """

        # Reset β, γ and μ to the values mentioned on Wikipedia (see https://bit.ly/2VMvb6h).

        Model.__DATA = None

        self.__beta = 0.4
        self.__gamma = 0.035
        self.__mu = 0.005

        # Reset I, R and D to the data at day 0 or the values mentioned on Wikipedia (see https://bit.ly/2VMvb6h).

        if self.__use_data:
            self.__x = np.array([self.__data_i(0), self.__data_r(0), self.__data_d(0)])
            self.__n = self.__population
        else:
            self.__x = np.array([3, 0, 0])
            self.__n = 1000

        # Reset our Unscented Kalman filter (if required). Note tat we use a dt value of 1 (day) and not the value of
        # Model.__DELTA_T.

        if self.__use_data:
            points = MerweScaledSigmaPoints(Model.__N_FILTERED,
                                            1e-3,  # Alpha value (usually a small positive value like 1e-3).
                                            2,  # Beta value (a value of 2 is optimal for a Gaussian distribution).
                                            0,  # Kappa value (usually, either 0 or 3-n).
                                            )

            self.__ukf = UnscentedKalmanFilter(Model.__N_FILTERED, Model.__N_MEASURED, 1, self.__h, Model.__f, points)

            self.__ukf.x = np.array([self.__data_i(0), self.__data_r(0), self.__data_d(0),
                                     self.__beta, self.__gamma, self.__mu, self.__n])

            self.__ukf.P *=15 

        # Reset our data (if requested).

        if self.__use_data:
            self.__data_s_values = np.array([self.__data_s(0)])
            self.__data_i_values = np.array([self.__data_i(0)])
            self.__data_r_values = np.array([self.__data_r(0)])
            self.__data_d_values = np.array([self.__data_d(0)])

        # Reset our predicted/estimated values.

        self.__s_values = np.array([self.__s_value()])
        self.__i_values = np.array([self.__i_value()])
        self.__r_values = np.array([self.__r_value()])
        self.__d_values = np.array([self.__d_value()])

        # Reset our estimated SIRD model parameters.

        self.__beta_values = np.array([self.__beta])
        self.__gamma_values = np.array([self.__gamma])
        self.__mu_values = np.array([self.__mu])

    def run(self, batch_filter=True, nb_of_days=100):
        """
        Run our SIRD model for the given number of days, taking advantage of the data (if requested) to estimate the
        values of β, γ and μ.
        """

        # Make sure that we were given a valid number of days.

        if not isinstance(nb_of_days, int) or nb_of_days <= 0:
            sys.exit('Error: \'nb_of_days\' must be an integer value greater than zero.')

        # Run our SIRD simulation, which involves computing our predicted/estimated state by computing our SIRD model /
        # Unscented Kalman filter in batch filter mode, if required.

        if self.__use_data and batch_filter:
            mu, cov = self.__ukf.batch_filter(self.__data)
            batch_filter_x, _, _ = self.__ukf.rts_smoother(mu, cov)

            # Override the first value of S, I, R and D.

            x = batch_filter_x[0][:3]

            self.__s_values = np.array([self.__n - x.sum()])
            self.__i_values = np.array([x[0]])
            self.__r_values = np.array([x[1]])
            self.__d_values = np.array([x[2]])

        for i in range(1, nb_of_days + 1):
            # Compute our predicted/estimated state by computing our SIRD model / Unscented Kalman filter for one day.

            if self.__use_data and self.__data_available(i):
                if batch_filter:
                    self.__x = batch_filter_x[i][:3]
                    self.__beta = batch_filter_x[i][3]
                    self.__gamma = batch_filter_x[i][4]
                    self.__mu = batch_filter_x[i][5]
                else:
                    self.__ukf.predict(model_self=self)
                    self.__ukf.update(np.array([self.__data_i(i), self.__data_r(i), self.__data_d(i)]))

                    self.__x = self.__ukf.x[:3]
                    self.__beta = self.__ukf.x[3]
                    self.__gamma = self.__ukf.x[4]
                    self.__mu = self.__ukf.x[5]
            else:
                for j in range(1, Model.__NB_OF_STEPS + 1):
                    self.__x = Model.__f(self.__x, Model.__DELTA_T, model_self=self, with_ukf=False)

            # Keep track of our data (if requested).

            if self.__use_data:
                if self.__data_available(i):
                    self.__data_s_values = np.append(self.__data_s_values, self.__data_s(i))
                    self.__data_i_values = np.append(self.__data_i_values, self.__data_i(i))
                    self.__data_r_values = np.append(self.__data_r_values, self.__data_r(i))
                    self.__data_d_values = np.append(self.__data_d_values, self.__data_d(i))
                else:
                    self.__data_s_values = np.append(self.__data_s_values, math.nan)
                    self.__data_i_values = np.append(self.__data_i_values, math.nan)
                    self.__data_r_values = np.append(self.__data_r_values, math.nan)
                    self.__data_d_values = np.append(self.__data_d_values, math.nan)

            # Keep track of our predicted/estimated values.

            self.__s_values = np.append(self.__s_values, self.__s_value())
            self.__i_values = np.append(self.__i_values, self.__i_value())
            self.__r_values = np.append(self.__r_values, self.__r_value())
            self.__d_values = np.append(self.__d_values, self.__d_value())

            # Keep track of our estimated SIRD model parameters.

            self.__beta_values = np.append(self.__beta_values, self.__beta)
            self.__gamma_values = np.append(self.__gamma_values, self.__gamma)
            self.__mu_values = np.append(self.__mu_values, self.__mu)

    def plot(self, figure=None, two_axes=False):
        """
        Plot the results using five subplots for 1) S, 2) I and R, 3) D, 4) β, and 5) γ and μ. In each subplot, we plot
        the data (if requested) as bars and the computed value as a line.
        """

        # days = range(self.__s_values.size)
        days = convert_to_date(self.__start_date, self.__s_values)
        nrows = 5 if self.__use_data else 3
        ncols = 1

        if figure is None:
            show_figure = True
            figure, axes = plt.subplots(nrows, ncols, figsize=Model.__FIG_SIZE, sharex=True)
        else:
            figure.clf()

            show_figure = False
            axes = figure.subplots(nrows, ncols, sharex=True)

        figure.canvas.set_window_title('SIRD model fitted to data' if self.__use_data else 'Wikipedia SIRD model')

        # First subplot: S.

        axis1 = axes[0]
        axis1.plot(days, self.__s_values, Model.__S_COLOR, label='S')
        axis1.legend(loc='best')
        if self.__use_data:
            axis2 = axis1.twinx() if two_axes else axis1
            axis2.bar(days, self.__data_s_values, color=Model.__S_COLOR, alpha=Model.__DATA_ALPHA)
            data_s_range = self.__population - min(self.__data_s_values)
            data_block = 10 ** (math.floor(math.log10(data_s_range)) - 1)
            s_values_shift = data_block * math.ceil(data_s_range / data_block)
            axis2.set_ylim(min(min(self.__s_values), self.__population - s_values_shift), self.__population)

        # Second subplot: I and R.

        axis1 = axes[1]
        axis1.plot(days, self.__i_values, Model.__I_COLOR, label='I')
        axis1.plot(days, self.__r_values, Model.__R_COLOR, label='R')
        axis1.legend(loc='best')
        if self.__use_data:
            axis2 = axis1.twinx() if two_axes else axis1
            axis2.bar(days, self.__data_i_values, color=Model.__I_COLOR, alpha=Model.__DATA_ALPHA)
            axis2.bar(days, self.__data_r_values, color=Model.__R_COLOR, alpha=Model.__DATA_ALPHA)

        # Third subplot: D.

        axis1 = axes[2]
        axis1.plot(days, self.__d_values, Model.__D_COLOR, label='D')
        axis1.legend(loc='best')
        if self.__use_data:
            axis2 = axis1.twinx() if two_axes else axis1
            axis2.bar(days, self.__data_d_values, color=Model.__D_COLOR, alpha=Model.__DATA_ALPHA)

        # Fourth subplot: β.

        if self.__use_data:
            axis1 = axes[3]
            axis1.plot(days, self.__beta_values, Model.__BETA_COLOR, label='β')
            axis1.legend(loc='best')

        # Fourth subplot: γ and μ.

        if self.__use_data:
            axis1 = axes[4]
            axis1.plot(days, self.__gamma_values, Model.__GAMMA_COLOR, label='γ')
            axis1.plot(days, self.__mu_values, Model.__MU_COLOR, label='μ')
            axis1.legend(loc='best')

        plt.xlabel('time (day)')

        if show_figure:
            plt.show()

    def movie(self, filename, batch_filter=True, nb_of_days=100):
        """
        Generate, if using the data, a movie showing the evolution of our SIRD model throughout time.
        """

        if self.__use_data:
            data_size = Model.__DATA.shape[0]
            figure = plt.figure(figsize=Model.__FIG_SIZE)
            backend = matplotlib.get_backend()
            writer = manimation.writers['ffmpeg']()

            matplotlib.use("Agg")

            with writer.saving(figure, filename, 96):
                for i in range(1, data_size + 1):
                    print('Processing frame #', i, '/', data_size, '...', sep='')

                    self.__data = Model.__DATA[:i]

                    self.reset()
                    self.run(batch_filter=batch_filter, nb_of_days=nb_of_days)
                    self.plot(figure=figure)

                    writer.grab_frame()

                print('All done!')

            matplotlib.use(backend)

    def s(self, day=-1):
        """
        Return all the S values (if day=-1) or its value for a given day.
        """

        if day == -1:
            return self.__s_values
        else:
            return self.__s_values[day]

    def i(self, day=-1):
        """
        Return all the I values (if day=-1) or its value for a given day.
        """

        if day == -1:
            return self.__i_values
        else:
            return self.__i_values[day]

    def r(self, day=-1):
        """
        Return all the R values (if day=-1) or its value for a given day.
        """

        if day == -1:
            return self.__r_values
        else:
            return self.__r_values[day]


    def d(self, day=-1):
        """
        Return all the D values (if day=-1) or its value for a given day.
        """

        if day == -1:
            return self.__d_values
        else:
            return self.__d_values[day]


    def beta(self, day=-1):
        """
        Return all the D values (if day=-1) or its value for a given day.
        """

        if day == -1:
            return self.__beta_values
        else:
            return self.__beta_values[day]


    def gamma(self, day=-1):
        """
        Return all the D values (if day=-1) or its value for a given day.
        """

        if day == -1:
            return self.__gamma_values
        else:
            return self.__gamma_values[day]

    def mu(self, day=-1):
        """
        Return all the D values (if day=-1) or its value for a given day.
        """

        if day == -1:
            return self.__mu_values
        else:
            return self.__mu_values[day]

    def days_array(self, day=-1):
        """
        Return all the D values (if day=-1) or its value for a given day.
        """
        days = convert_to_date(self.__start_date, self.__s_values)
        return days
    def days_cases(self, day=-1):
        """
        Return all the D values (if day=-1) or its value for a given day.
        """
        days = convert_to_date(self.__start_date, self.__confirmed_data)
        return days

    def confirmed_data(self, day=-1):
        """
        Return all the D values (if day=-1) or its value for a given day.
        """
        return self.__confirmed_data
    def recovered_data(self, day=-1):
        """
        Return all the D values (if day=-1) or its value for a given day.
        """
        return self.__recovered_data
    def deaths_data(self, day=-1):
        """
        Return all the D values (if day=-1) or its value for a given day.
        """
        return self.__deaths_data
    def active_data(self, day=-1):
        """
        Return all the D values (if day=-1) or its value for a given day.
        """
        return self.__confirmed_data - self.__recovered_data - self.__deaths_data

def age_range_plot(xdata, ydata, distribution, age_range):
    scale_up = 0
    for k in range(len(distribution)):
        if(k==0):
            scale_down = 0
        else:
            scale_down = scale_down + distribution[k-1]
        scale_up = scale_up + distribution[k]

        y_down =  ydata*scale_down/100
        y_up =  ydata*scale_up/100
        plt.fill_between(xdata, y_down, y_up, label='Age: '+age_range[k]+' Year', alpha=0.4)

if __name__ == '__main__':
    # Create an instance of the SIRD model, asking for the data to be used.

    countries = ['Bangladesh']
    run_time = [120, 140, 115]
    population = 170.1e6
    ### Bangladesh susceptible population 5,26,73,000

    age_range = [ '1 - 10', '11 - 20', '21 - 30', '31 - 40', '41 - 50', '51 - 60', '60+' ]
    age_case_distribution = np.array([ 2.9, 7.3, 27.6, 27.1, 17.3, 11.1, 6.7 ])
    age_death_distribution = np.array([ 1.01, 1.85, 3.52, 8.05, 17.11, 31.38, 37.08 ])
    divisor = 1e3

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # colors = ['tab:red', 'tab:blue']
    # plt.setp(ax1.xaxis.get_majorticklabels(),rotation=45,horizontalalignment='right')
    # markers = ['-', '-.', '--']
    tag = 0
    for country in countries:
        m = Model(country=country)

        m.reset()

        # Run our SIRD model and plot its S, I, R and D values, together with the data.

        m.run(nb_of_days=run_time[tag])
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

        data = {
                'days': days_case,
                'confirmed': m.confirmed_data(),
                'recovered': m.recovered_data(),
                'deaths': m.deaths_data(),
                'active': m.active_data(),
                'days_sim': days_array,
                'susceptible_sim': m.s(),
                'active_sim': m.i(),
                'recovered_sim': m.r(),
                'deaths_sim': m.d(),
                'beta': m.beta(),
                'gamma': m.gamma(),
                'mu': m.mu(),
                'r0': m.beta()/(m.gamma() + m.mu())
                }

        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))
        df.to_csv(country+'.csv', index=False)

        # plt.figure(2)
        # plt.plot(days_array, m.i()/divisor, label=country)
        # plt.plot(days_case, m.active_data()/divisor, 'kx')
        # age_range_plot(days_array, m.i()/divisor, age_case_distribution, age_range)
        # plt.ylabel('Active cases (x'+str(int(divisor))+')')
        # plt.xlabel('Date')
        # plt.legend()
        # plt.xticks(rotation=45)

        # plt.figure(3)
        # plt.plot(days_array, m.d()*10/divisor, label=country)
        # plt.plot(days_case, m.deaths_data()*10/divisor, 'kx')
        # age_range_plot(days_array, m.d()*10/divisor, age_death_distribution, age_range)
        # plt.ylabel('Death cases (x'+str(int(divisor/10))+')')
        # plt.xlabel('Date')
        # plt.legend()
        # plt.xticks(rotation=45)

        # # ax1.plot(days_case, m.recovered_data()/divisor, 'kx')
        # # ax1.plot(days_array, m.r()/divisor, linestyle=markers[tag], color=colors[0])
        # # ax1.set_ylabel('Recovered cases (x'+str(int(divisor))+')', color=colors[0])
        # ax1.plot(days_case, m.deaths_data()*100/m.recovered_data(), 'r.')
        # ax1.plot(days_array, m.d()*100/m.r(), linestyle=markers[tag], color=colors[0])
        # ax1.set_ylabel('CFR for Recovered Cases (in %)', color=colors[0])
        # ax1.tick_params(axis='y', labelcolor=colors[0])
        # ax1.set_xlabel('Date')

        # ax2.plot(days_case, m.deaths_data()*100/m.confirmed_data(), 'b.')
        # ax2.plot(days_array, m.d()*100/(m.r()+m.i()+m.d()), linestyle=markers[tag], color=colors[1])
        # ax2.set_ylabel('CFR for Confirmed Cases (in %)', color=colors[1])
        # ax2.tick_params(axis='y', labelcolor=colors[1])
        # fig.tight_layout()

        # ### Estimated value for 10 days
        # start_length = len(m.confirmed_data())
        # for estimated_day in range(start_length, start_length+10):
        #     total_deaths = m.d(estimated_day)
        #     total_recover = m.r(estimated_day) 
        #     total = total_recover + m.i(estimated_day) + total_deaths
        #     print('Date: ', days_array[estimated_day], ' Cases: ', int(total), '$\pm$', int(total/100), ' Recover: ', int(total_recover), '$\pm$', int(total_recover/100), ' Deaths: ', int(total_deaths), '$\pm$', int(total_deaths/100))

        # plt.figure(4)
        # r0 = m.beta()/(m.gamma() + m.mu())
        # ## basic reproduction number
        # plt.plot(days_array, r0, label='$R_0$')
        # plt.plot(days_array, r0*m.s()/m.s(0), label='$R_e$')

        # ## force of infection
        # # plt.plot(days_array, m.beta()*m.i(), label='R')

        # plt.ylabel('Magnitude of Reproduction Number')
        # plt.xlabel('Date')
        # plt.legend()
        # plt.xticks(rotation=45)
        # plt.ylim(0, 20)

        
        # fig, ax = plt.subplots()
        # ax.plot(days_array, m.i()/(divisor), label=country)
        # ax.plot(days_case, m.active_data()/divisor, 'kx')

        # ax.set_ylabel('Active cases (x'+str(int(divisor))+')')
        # ax.set_xlabel('Date')
        # ax.legend()
        # plt.xticks(rotation=45)

        # def number2percent(y):
        #     return y * 100.0 *divisor/ population

        # def percent2number(y):
        #     return y * population/(divisor* 100.0)

        # secax = ax.secondary_yaxis('right', functions=(number2percent, percent2number))
        # secax.set_ylabel('Percent of the population')

        tag = tag + 1

    plt.show()
