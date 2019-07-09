import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import pmdarima as pm
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

df_orig = pd.read_csv('.\Market_food_prices_w_temp_reduced.csv')
country_dict = dict(zip(df_orig.country_id,df_orig.country))
plt.rcParams["figure.figsize"] = [16, 12]

text_file1 = open(".\Auto-ARIMA_modeling_summary.doc", "w")

#fitting using Auto-ARIMA for food price
def auto_arima_prices(df_country, value, log):
    print('summary using auto-ARIMA for prices...')
    print('Auto-ARIMA results for food prices', file=text_file1)
    plt.figure()
    df_prices = df_country.groupby('period')['price_in_USD'].mean().reset_index()
    df_prices.set_index('period', inplace=True)
    df_prices=df_prices.sort_index()

    #log transformation with differencing of 1
    if (log=='Y'):
        print('log transformation...')
        df_trans = df_prices.copy()
        df_trans['price_in_USD'] = np.log(df_trans.iloc[:, 0]).diff(1)
        df_trans = df_trans.dropna()
        X = df_trans
    else:
        X = df_prices

    #get dynamic size of split based on time series
    init = int(len(X) * 0.75) #get the 75% first
    if (init%12==0):          #check if it has the full year
        size = init
    else:
        size = int(init/12)*12 + 12 #calculate the next full year split

    print('size:' + str(size))
    train, valid = X[0:size], X[size:len(X)]

    model = pm.auto_arima(train, start_p=1, start_q=1,
                          test='adf',       # use adftest to find optimal 'd'
                          max_p=3, max_q=3, # maximum p and q
                          d=None,           # let model determine 'd'
                          seasonal=False,   # No Seasonality
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

    print(model.summary(), file=text_file1)
    model.fit(train)

    forecast = model.predict(n_periods=len(valid))
    forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])

    plt.plot(train, label='Train', color='blue')
    plt.plot(valid, label='Validation', color='seagreen')
    plt.plot(forecast, label='Prediction', color='red')
    plt.title('Auto Arima Modeling')
    plt.xlabel('period')
    plt.ylabel('food prices')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('./plot-auto_arima/foodprices/'+ value + '.png')

    #calculate rmse
    rms = sqrt(mean_squared_error(valid, forecast))
    print('RMSE: ' + str(rms), file=text_file1)
    print('='*70, file=text_file1)

#fitting using Auto-ARIMA for avg temp
def auto_arima_temp(df_country, value, log):
    print('summary using auto-ARIMA for avg_temp...')
    print('Auto-ARIMA results for avg_temp', file=text_file1)
    plt.figure()
    df_temp = df_country.groupby('period')['avg_temp'].mean().reset_index()
    df_temp.set_index('period', inplace=True)
    df_temp=df_temp.sort_index()

    #log transformation with differencing of 1
    if (log == 'Y'):
        print('log transformation...')
        df_trans = df_temp.copy()
        df_trans['avg_temp'] = np.log(df_trans.iloc[:, 0]).diff(1)
        df_trans = df_trans.dropna()
        X = df_trans.values
    else:
        X = df_temp

    #get dynamic size of split based on time series
    init = int(len(X) * 0.75) #get the 75% first
    if (init%12==0):          #check if it has the full year
        size = init
    else:
        size = int(init/12)*12 + 12 #calculate the next full year split

    train, valid = X[0:size], X[size:len(X)]

    model = pm.auto_arima(train, start_p=1, start_q=1,
                          test='adf',       # use adftest to find optimal 'd'
                          max_p=3, max_q=3, # maximum p and q
                          d=None,           # let model determine 'd'
                          seasonal=False,   # No Seasonality
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

    print(model.summary(), file=text_file1)
    model.fit(train)

    forecast = model.predict(n_periods=len(valid))
    forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])

    plt.plot(train, label='Train', color='blue')
    plt.plot(valid, label='Validation', color='seagreen')
    plt.plot(forecast, label='Prediction', color='red')
    plt.title('Auto Arima Modeling')
    plt.xlabel('period')
    plt.ylabel('avg temp')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('./plot-auto_arima/avgtemp/'+ value + '.png')

    #calculate rmse
    rms = sqrt(mean_squared_error(valid, forecast))
    print('RMSE: ' + str(rms), file=text_file1)
    print('-'*70, file=text_file1)

for key, value in country_dict.items():
    df_country = pd.read_csv('.\countries\country_df_'+ value +'.csv')
    df_country.sort_values(by='period')
    df_country['period'] = pd.to_datetime(df_country.period, format='%Y-%m-%d')
    drop_cols = ['country_id', 'country', 'commodity_purchased']
    df_country.drop(drop_cols, axis=1, inplace=True)
    print('='*70, file=text_file1)
    print(str(value), file=text_file1)
    print('='*70, file=text_file1)
    print('='*70)
    print(str(value))
    print('='*70)
    auto_arima_temp(df_country, value, 'N') #df, country name, log=Y if log and differencing to be performed
    auto_arima_prices(df_country, value, 'N') #df, country name, log=Y if log and differencing to be performed
