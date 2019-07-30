import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas import DataFrame
from pandas import concat
import numpy as np
from pandas import Series

def stationary_check(df, country, col, path, output, type):
    timeseries = df[col]

    rolmean = timeseries.rolling(12).mean() ## as month is year divide by 12
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(16,12))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std  = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Std Deviation')

    #ADF test
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    print("=" * 70, file=output)
    print(country, file=output)
    print("="*70, file=output)
    print ('Dickey-Fuller Test for ' +col , file=output)
    print (dfoutput, file=output)
    print('Critical Values:', file=output)
    for key, value in dftest[4].items():
         print('\t%s: %.3f' % (key, value), file=output)
         dfoutput['Critical Value (%s)'%key] = value
    #cutoff=0.01

    adf_check = 0
    if dfoutput['Test Statistic'] > dfoutput['Critical Value (5%)']:
        #accept null hypothesis (series has a unit root)
        print('p-value = ' + str(dftest[1])+ '\nThe series ' + timeseries.name + ' is likely non-stationary', file=output)
        adf_check = 1
    else:
        #reject null hypothesis (series has no unit root)
        print('p-value = ' + str(dftest[1]) + '\nThe series ' + timeseries.name + ' is likely stationary', file=output)
    print("="*70, file=output)

    plt.figtext(.3,.15,str(dfoutput))
    plt.savefig(path + country + '.png')

    #kpss test
    kpss_check = 0
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
       kpss_output['Critical Value (%s)'%key] = value
    print ('KPSS Test for ' +col , file=output)
    print(kpss_output, file= output)
    if kpss_output['Test Statistic'] > kpss_output['Critical Value (5%)']:
        #reject the null hypothesis (series has a unit root - non-stationary)
        print('p-value = ' + str(dftest[1])+ '\nThe series ' + timeseries.name + ' is likely non-stationary', file=output)
    else:
        #accept the null hypothesis (has trend stationarity)
        print('p-value = ' + str(dftest[1]) + '\nThe series ' + timeseries.name + ' is likely stationary', file=output)
        kpss_check = 1

    if(kpss_check ==  0) and (adf_check == 1):
        # Unit root: accept null; KPSS test: reject null
        print('***Both imply that series has unit root (non-stationary)', file=output)
    elif (kpss_check == 1) and (adf_check == 0):
        #unit root: reject null; KPSS test: accept null
        print('***Both imply that series is stationary', file=output)
    elif (kpss_check == 1) and (adf_check == 1):
        # unit root: reject null; KPSS test: reject null
        print('***ADF - no unit root but KPSS has a unit root - need to check further', file=output)
    elif (kpss_check == 0) and (adf_check == 0):
        # unit root: accept null; KPSS test: accept null
        print('***The process is trend-stationary but has a unit root', file=output)

def plot_acf_pacf(df, country, path):
    plt.figure()
    fig, (ax1,ax2) = plt.subplots(2, 1)
    plot_acf(df, lags=50, marker='o', ax=ax1)
    plt.title('Autocorrelation Function')
    plt.xlabel('number of lags')
    plt.ylabel('correlation')

    plot_pacf(df, lags=50, marker='o', ax=ax2)
    plt.title('Partial Autocorrelation Function')
    plt.xlabel('number of lags')
    plt.ylabel('correlation')
    fig = plt.gcf()
    fig.savefig('./'+ path + '/merged_acf/'+ country + '.png')

# convert series to supervised learning
def series_to_supervise(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#get the dynamic size of the split
def split_data(df_trans, sort):
    if (sort=='Y'):
       df_trans = df_trans.sort_values(by='period')

    init = int(len(df_trans) * 0.75) #get the 75% first
    if (init%12==0):          #check if it has the full year
        size = init
    else:
        size = int(init/12)*12 + 12 #calculate the next full year split

    train, valid = df_trans[0:size], df_trans[size:len(df_trans)]
    return train, valid

def reverse_log(df_log):
    df_reverse = np.exp(df_log)
    return df_reverse

def undiff(dataset, interval):
    undiff = []
    for i in range(0, interval):
        value=dataset[i]
        undiff.append(value)
    for i in range(0, len(dataset)-interval):
        value = undiff[i] + dataset[i + interval]
        undiff.append(value)
    return Series(undiff)

