import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from my_functions import stationary_check, plot_acf_pacf, series_to_supervise, split_data, reverse_log,undiff
import pmdarima as pm
from scipy.stats import boxcox
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from scipy.special import  inv_boxcox

plt.rcParams["figure.figsize"] = [16, 12]

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def percentage_error_to_mean(y_true_mean, rmse):
    pem = (rmse/y_true_mean)
    return pem * 100

def make_stationary(df_country, value, log, split):
    df_country['period'] = pd.to_datetime(df_country.period, format='%Y-%m-%d')
    df_country.set_index('period', inplace=True)
    df_country.sort_values(by='period')
    df_trans_orig = df_country.copy()
    df_log = df_trans_orig.copy()
    if (log == 'T0'): #El Salvador
        df_trans_log = df_trans_orig.copy()
        df_log = np.log(df_trans_orig)
        df_trans_log ['price_in_USD'] = np.log(df_trans_orig ['price_in_USD'])
        df_trans_log['avg_temp'] = np.log(df_trans_orig['avg_temp']).diff(5) #from 2 to 1
        movingAverage = df_trans_log['avg_temp'].rolling(window=12).mean()
        movingAverage2 = df_trans_log['price_in_USD'].rolling(window=12).mean()
        df_trans_log['avg_temp'] = df_trans_log['avg_temp'] - movingAverage
        df_trans_log['price_in_USD'] = df_trans_log['price_in_USD'] - movingAverage2
        df_trans = df_trans_log.dropna()
    elif (log == 'T1'): #Tajikistan, Costa RIca, Bangladesh, Central African Republic, Chad
        df_trans_log = df_trans_orig.copy()
        df_log = np.log(df_trans_orig)
        df_trans_log = np.log(df_trans_orig).diff(1)
        df_trans = df_trans_log.dropna()
    elif (log == 'T2'): #  'Mali', 'Senegal', 'Benin', 'Burundi', 'Chad', 'Guatemala', 'Niger'
        df_trans_log = df_trans_orig.copy()
        df_log = np.log(df_trans_orig)
        df_trans_log = np.log(df_trans_orig).diff(1)
        movingAverage = df_trans_log.rolling(window=12).mean()
        df_trans = df_trans_log - movingAverage
        df_trans = df_trans.dropna()
    elif (log == 'T3'):#India
        df_trans_log = df_trans_orig.copy()
        df_log = np.log(df_trans_orig)
        df_trans_log = np.log(df_trans_orig).diff(3)
        movingAverage = df_trans_log.rolling(window=12).mean()
        df_trans = df_trans_log - movingAverage
        df_trans = df_trans.dropna()
    elif (log == 'T4'):#Mozambique
        df_trans_log = df_trans_orig.copy()
        df_log = np.log(df_trans_orig)
        df_trans_log['avg_temp'] = np.log(df_trans_orig['avg_temp']).diff(1)
        df_trans_log ['price_in_USD'] = np.log(df_trans_orig ['price_in_USD']).diff(5)
        df_trans = df_trans_log.dropna()
    elif (log == 'T5'): # Nepal
        df_trans_log = df_trans_orig.copy()
        df_log = np.log(df_trans_orig)
        df_trans_log ['price_in_USD'] = np.log(df_trans_orig ['price_in_USD']).diff(1)
        df_trans_log['avg_temp'] = np.log(df_trans_orig['avg_temp']).diff(2)
        movingAverage = df_trans_log['avg_temp'].rolling(window=12).mean()
        movingAverage2 = df_trans_log['price_in_USD'].rolling(window=12).mean()
        df_trans_log['avg_temp'] = df_trans_log['avg_temp'] - movingAverage
        df_trans_log['price_in_USD'] = df_trans_log['price_in_USD'] - movingAverage2
        df_trans = df_trans_log.dropna()
    elif (log == 'T6'): #[ 'Kenya']
        print('log based difference with diff=1 on price and 3 on avg temp...')
        df_trans_log = df_trans_orig.copy()
        df_log = np.log(df_trans_orig)
        df_trans_log['avg_temp'] = np.log(df_trans_orig['avg_temp']).diff(1) #from 1
        df_trans_log ['price_in_USD'] = np.log(df_trans_orig ['price_in_USD']).diff(1)
        movingAverage2 = df_trans_log['price_in_USD'].rolling(window=12).mean()
        df_trans_log['price_in_USD'] = df_trans_log['price_in_USD'] - movingAverage2
        df_trans = df_trans_log.dropna()
    elif (log == 'T7'): #[ 'Peru']
        print('log based difference with diff=1 on price and 1 on avg temp...')
        df_trans_log = df_trans_orig.copy()
        df_log = np.log(df_trans_orig)
        df_trans_log['avg_temp'] = np.log(df_trans_orig['avg_temp'])
        df_trans_log ['price_in_USD'] = np.log(df_trans_orig ['price_in_USD']).diff(1)
        movingAverage2 = df_trans_log['price_in_USD'].rolling(window=12).mean()
        df_trans_log['price_in_USD'] = df_trans_log['price_in_USD'] - movingAverage2
        df_trans = df_trans_log.dropna()
    elif (log == 'T8'): #Guatemala
        print('log transformation with diff only')
        df_trans_log = df_trans_orig.copy()
        df_log = np.log(df_trans_orig)
        df_trans_log['avg_temp'] = np.log(df_trans_orig['avg_temp']).diff(3)
        df_trans_log ['price_in_USD'] = np.log(df_trans_orig ['price_in_USD']).diff(1)
        df_trans = df_trans_log.dropna()
    else:
        df_trans = df_trans_orig.copy()

    #check only for training set
    if (split == 'train'):
        # check stationarity after transformation
        stationary_check(df_trans, value, 'avg_temp', './plot-var-model/transformed/avgtemp/adf/', text_file2, 'temp')
        stationary_check(df_trans, value, 'price_in_USD', './plot-var-model/transformed/foodprices/adf/', text_file3,
                         'prices')
        # plot_acf/pacf - prices
        df_prices_train = df_trans.groupby('period')['price_in_USD'].mean().reset_index()
        df_prices_train.set_index('period', inplace=True)
        plot_acf_pacf(df_prices_train, value, 'plot-var-model/transformed/foodprices')
        # plot_acf/pacf - temp
        df_temp_train = df_trans.groupby('period')['avg_temp'].mean().reset_index()
        df_temp_train.set_index('period', inplace=True)
        plot_acf_pacf(df_temp_train, value, 'plot-var-model/transformed/avgtemp')

    return df_trans, df_log

def auto_arima_model(train, test, value, path, filename, sea_val, full, difftype, lam):
    print('='*80, file=filename)
    print(str(value), file=filename)
    print('='*80, file=filename)
    model_arima = pm.auto_arima(train, start_p=1, start_q=1,
                          test='adf',
                          max_p=5, max_q=5,
                          d=None,
                          seasonal=sea_val,
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

    print(model_arima.summary(), file=filename)

    model_arima.fit(train)
    forecast = model_arima.predict(n_periods=len(test))
    forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])

    #reverse based on original values
    if (difftype =='log'):
        train_orig = reverse_log(train)
        test_orig = reverse_log(test)
        forecast_orig = reverse_log(forecast)
    elif (difftype == 'bc'):
        train_orig = inv_boxcox(train, lam)
        test_orig = inv_boxcox(test, lam)
        forecast_orig = inv_boxcox(forecast, lam)
    else:
        train_orig = train.copy()
        test_orig = test.copy()
        forecast_orig = forecast.copy()

    rmse = sqrt(mean_squared_error(test_orig, forecast_orig))
    mape = mean_absolute_percentage_error(test_orig, forecast_orig)
    pce = percentage_error_to_mean(np.mean(test_orig)[0], rmse)

    #plot
    plt.figure()
    plt.figtext(.2,.15, 'rmse=%f' % (rmse))
    plt.plot(test_orig, label='test set', color='blue')
    plt.plot(forecast_orig, label='prediction', color='red')
    plt.title('Auto Arima Modeling - Actual vs Predict ' + value)
    plt.xlabel('period')
    plt.ylabel(path)
    plt.legend()
    fig = plt.gcf()
    fig.savefig('./plot-auto-arima/transformed/'+ path +'/'+ value + '.png')

    test_df = test.stack().reset_index()
    test_df.columns = ['period', 'name', 'price']

    # for predicting future 36 months using the full dataset
    model_arima.fit(full)
    date_rng = pd.date_range(start=test_df.iloc[-1,0], periods=36, freq='M')
    future = model_arima.predict(n_periods=36)
    future = pd.DataFrame(future, index=date_rng, columns=['Future'])

    plt.figure()
    plt.plot(future, label='future', color='orange')
    plt.plot(full, label='past', color='blue')
    plt.title('Auto Arima Modeling - Future ' + value)
    plt.xlabel('period')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('./plot-auto-arima/transformed/'+ path +'/future/'+ value + '.png')

    print('RMSE: ' + str(rmse), file=filename)
    print('MAPE: ' + str(mape), file=filename)
    print('-'*80, file=filename)

    #reverse based on original values
    if (difftype =='log'):
        future_orig = reverse_log(future)
        full_orig = reverse_log(full)
    elif (difftype == 'bc'):
        future_orig = inv_boxcox(future,lam)
        full_orig = inv_boxcox(full,lam)
    else:
        future_orig = future
        full_orig = full

    #plot based on original values
    if (path == 'avgtemp'):
        mean_ARIMA = future_orig['Future'].mean() - full_orig['avg_temp'].mean()
    else:
        mean_ARIMA = future_orig['Future'].mean() - full_orig['price_in_USD'].mean()
    plt.figure()
    plt.figtext(.2,.15, 'mean diff= ' + str(mean_ARIMA))
    plt.plot(future_orig, label='future', color='orange')
    plt.plot(full_orig, label='past', color='blue')
    plt.title('Auto Arima Modeling - Future ' + value)
    plt.xlabel('period')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('./plot-auto-arima/transformed/'+ path +'/future/origval/'+ value + '.png')

    return mean_ARIMA, rmse

#fitting the model using VAR
def var_model(train, test, value, full, difftype_price, df_log, train_before, test_before):
    model_var = VAR(endog=train, exog=train['avg_temp'])
    # model_var = VAR(endog=train)
    model_fit = model_var.fit()
    print(model_fit.summary(), file=text_file1)

    cols = train.columns
    prediction = model_fit.forecast(model_fit.y, steps=len(test), exog_future=test['avg_temp'])
    # prediction = model_fit.forecast(model_fit.y, steps=len(test))

    #converting predictions to dataframe
    pred = pd.DataFrame(index=range(0, len(prediction)), columns=[cols])
    for j in range(0, 2):
        for i in range(0, len(prediction)):
            pred.iloc[i][j] = prediction[i][j]

    test = test.reset_index()
    pred= pred.reset_index()
    pred['period'] = test['period']
    pred.columns=['sno', 'avg_temp', 'price_in_USD',  'period']
    test = test.set_index('period')
    pred = pred.set_index('period')
    pred = pred.drop('sno', 1)

    #initialize
    pred_orig = pd.DataFrame(index=pred.index)
    train_orig = pd.DataFrame(index=train.index)
    test_orig = pd.DataFrame(index=test.index)
    train_orig['price_in_USD'] = train['price_in_USD'].astype('float32')
    pred_orig['price_in_USD'] = pred['price_in_USD'].astype('float32')
    test_orig['price_in_USD'] = test['price_in_USD'].astype('float32')

    #inverse-transform before plotting
    if (difftype_price =='log'):
        train_orig ['price_in_USD'] = reverse_log(train_orig['price_in_USD'] )
        pred_orig ['price_in_USD']  = reverse_log(pred_orig ['price_in_USD'])
        test_orig ['price_in_USD']  = reverse_log(test_orig ['price_in_USD'])

    elif (difftype_price =='logb'):
        movingAvg1 =  test_before['price_in_USD'].diff(1).rolling(window=12).mean()
        test_orig['price_in_USD'] = test_orig['price_in_USD'] + movingAvg1
        movingAvg2 =  train_before['price_in_USD'].diff(1).rolling(window=12).mean()
        train_orig['price_in_USD'] = train_orig['price_in_USD'] + movingAvg2

        #reverse future values
        pred_orig = pred_orig.reset_index()
        pred_orig['price_in_USD'] = reverse_log(pred_orig['price_in_USD'])

        #reverse past values
        train_orig = train_orig.reset_index()
        train_orig['price_in_USD'] = reverse_log(train_orig['price_in_USD'])

        #reverse test values
        test_orig = test_orig.reset_index()
        test_orig['price_in_USD'] = reverse_log(test_orig['price_in_USD'])

        #print('before plotting')
        train_orig = train_orig.set_index('period')
        pred_orig = pred_orig.set_index('period')
        test_orig = test_orig.set_index('period')

    elif (difftype_price == 'logD1'):
        val_train = train_before.head(1)['price_in_USD']
        val_pred = train_before.tail(1)['price_in_USD']
        val_test = test_before.head(1)['price_in_USD']

        new_series_train = pd.Series(val_train).append(pd.Series(train_orig['price_in_USD']))
        new_series_train.columns = ['price_in_USD']
        new_series_pred = pd.Series(val_pred).append(pd.Series(pred_orig['price_in_USD']))
        new_series_pred.columns = ['price_in_USD']
        new_series_test = pd.Series(val_test).append(pd.Series(test_orig['price_in_USD']))
        new_series_test.columns = ['price_in_USD']

        #reverse future values
        train_orig = train_orig.reset_index()
        train_orig['price_in_USD'] = undiff(new_series_train, 1)

        train_orig['price_in_USD'] = reverse_log(train_orig['price_in_USD'])

        #reverse past values
        pred_orig = pred_orig.reset_index()
        pred_orig['price_in_USD'] = undiff(new_series_pred, 1)
        pred_orig['price_in_USD'] = reverse_log(pred_orig['price_in_USD'])

        #reverse test values
        test_orig = test_orig.reset_index()
        test_orig['price_in_USD'] = undiff(new_series_test, 1)
        test_orig['price_in_USD'] = reverse_log(test_orig['price_in_USD'])

        train_orig = train_orig.set_index('period')
        pred_orig = pred_orig.set_index('period')
        test_orig = test_orig.set_index('period')

    elif (difftype_price == 'logD2'):
        val_train = train_before.head(2)['price_in_USD']
        val_pred = train_before.tail(2)['price_in_USD']
        val_test = test_before.head(2)['price_in_USD']

        new_series_train = pd.Series(val_train).append(pd.Series(train_orig['price_in_USD']))
        new_series_train.columns = ['price_in_USD']
        new_series_pred = pd.Series(val_pred).append(pd.Series(pred_orig['price_in_USD']))
        new_series_pred.columns = ['price_in_USD']
        new_series_test = pd.Series(val_test).append(pd.Series(test_orig['price_in_USD']))
        new_series_test.columns = ['price_in_USD']

        #reverse future values
        train_orig = train_orig.reset_index()
        train_orig['price_in_USD'] = undiff(new_series_train, 2)
        train_orig['price_in_USD'] = reverse_log(train_orig['price_in_USD'])

        #reverse past values
        pred_orig = pred_orig.reset_index()
        pred_orig['price_in_USD'] = undiff(new_series_pred, 2)
        pred_orig['price_in_USD'] = reverse_log(pred_orig['price_in_USD'])

        #reverse test values
        test_orig = test_orig.reset_index()
        test_orig['price_in_USD'] = undiff(new_series_test, 2)
        test_orig['price_in_USD'] = reverse_log(test_orig['price_in_USD'])

        train_orig = train_orig.set_index('period')
        pred_orig = pred_orig.set_index('period')
        test_orig = test_orig.set_index('period')

    elif (difftype_price == 'logD5'):

        val_pred = train_before.tail(5)['price_in_USD']
        val_train = train_before.head(5)['price_in_USD']
        val_test = test_before.head(5)['price_in_USD']

        new_series_pred = pd.Series(val_pred).append(pd.Series(pred_orig['price_in_USD']))
        new_series_pred.columns = ['price_in_USD']
        new_series_train = pd.Series(val_train).append(pd.Series(train_orig['price_in_USD']))
        new_series_train.columns = ['price_in_USD']
        new_series_test = pd.Series(val_test).append(pd.Series(test_orig['price_in_USD']))
        new_series_test.columns = ['price_in_USD']

        #reverse future values
        pred_orig = pred_orig.reset_index()
        pred_orig['price_in_USD'] = undiff(new_series_pred, 5)
        pred_orig['price_in_USD'] = reverse_log(pred_orig['price_in_USD'])

        #reverse past values
        train_orig = train_orig.reset_index()
        train_orig['price_in_USD'] = undiff(new_series_train, 5)
        train_orig['price_in_USD'] = reverse_log(train_orig['price_in_USD'])

        #reverse test values
        test_orig = test_orig.reset_index()
        test_orig['price_in_USD'] = undiff(new_series_test, 5)
        test_orig['price_in_USD'] = reverse_log(test_orig['price_in_USD'])

        train_orig = train_orig.set_index('period')
        pred_orig = pred_orig.set_index('period')
        test_orig = test_orig.set_index('period')

    elif (difftype_price == 'logbD1'):
        val_pred = train_before.tail(1)['price_in_USD']
        val_train = train_before.head(1)['price_in_USD']
        val_test = test_before.head(1)['price_in_USD']

        movingAvg1 =  test_before['price_in_USD'].diff(1).rolling(window=12).mean()
        test_orig['price_in_USD'] = test_orig['price_in_USD'] + movingAvg1
        movingAvg2 =  train_before['price_in_USD'].diff(1).rolling(window=12).mean()
        train_orig['price_in_USD'] = train_orig['price_in_USD'] + movingAvg2

        new_series_pred = pd.Series(val_pred).append(pd.Series(pred_orig['price_in_USD']))
        new_series_pred.columns = ['price_in_USD']
        new_series_train = pd.Series(val_train).append(pd.Series(train_orig['price_in_USD']))
        new_series_train.columns = ['price_in_USD']
        new_series_test = pd.Series(val_test).append(pd.Series(test_orig['price_in_USD']))
        new_series_test.columns = ['price_in_USD']

        #reverse future values
        pred_orig = pred_orig.reset_index()
        pred_orig['price_in_USD'] = undiff(new_series_pred, 1)
        pred_orig['price_in_USD'] = reverse_log(pred_orig['price_in_USD'])

        #reverse past values
        train_orig = train_orig.reset_index()
        train_orig['price_in_USD'] = undiff(new_series_train, 1)
        train_orig['price_in_USD'] = reverse_log(train_orig['price_in_USD'])

        #reverse test values
        test_orig = test_orig.reset_index()
        test_orig['price_in_USD'] = undiff(new_series_test, 1)
        test_orig['price_in_USD'] = reverse_log(test_orig['price_in_USD'])

        train_orig = train_orig.set_index('period')
        pred_orig = pred_orig.set_index('period')
        test_orig = test_orig.set_index('period')

    elif (difftype_price == 'logbD3'):
        val_pred = train_before.tail(3)['price_in_USD']
        val_train = train_before.head(3)['price_in_USD']
        val_test = test_before.head(3)['price_in_USD']

        movingAvg1 =  test_before['price_in_USD'].diff(3).rolling(window=12).mean()
        test_orig['price_in_USD'] = test_orig['price_in_USD'] + movingAvg1
        movingAvg2 = train_before['price_in_USD'].diff(3).rolling(window=12).mean()
        train_orig['price_in_USD'] = train_orig['price_in_USD'] + movingAvg2

        new_series_pred = pd.Series(val_pred).append(pd.Series(pred_orig['price_in_USD']))
        new_series_pred.columns = ['price_in_USD']
        new_series_train = pd.Series(val_train).append(pd.Series(train_orig['price_in_USD']))
        new_series_train.columns = ['price_in_USD']
        new_series_test = pd.Series(val_test).append(pd.Series(test_orig['price_in_USD']))
        new_series_test.columns = ['price_in_USD']

        # reverse future values
        pred_orig = pred_orig.reset_index()
        pred_orig['price_in_USD'] = undiff(new_series_pred, 3)
        pred_orig['price_in_USD'] = reverse_log(pred_orig['price_in_USD'])

        # reverse past values
        train_orig = train_orig.reset_index()
        train_orig['price_in_USD'] = undiff(new_series_train, 3)
        train_orig['price_in_USD'] = reverse_log(train_orig['price_in_USD'])

        #reverse test values
        test_orig = test_orig.reset_index()
        test_orig['price_in_USD'] = undiff(new_series_test, 3)
        test_orig['price_in_USD'] = reverse_log(test_orig['price_in_USD'])

        train_orig = train_orig.set_index('period')
        pred_orig = pred_orig.set_index('period')
        test_orig = test_orig.set_index('period')
    else:
        pred_orig ['price_in_USD'] = pred ['price_in_USD']
        train_orig  ['price_in_USD']  = train ['price_in_USD']
        test_orig['price_in_USD'] = test['price_in_USD']

    #check errors
    rmse = sqrt(mean_squared_error(test_orig, pred_orig))
    mape = mean_absolute_percentage_error(test_orig, pred_orig)
    pce = percentage_error_to_mean(np.mean(test_orig)[0], rmse)
    print('RMSE value for food prices is : ', sqrt(mean_squared_error(test_orig, pred_orig)), file=text_file1)
    print('MAE value for food prices is : ', mean_absolute_error(test_orig, pred_orig), file=text_file1)
    print('MAPE value for food prices is : ', mean_absolute_percentage_error(test_orig, pred_orig), file=text_file1)

    #rename
    pred_orig.columns=['price_in_USD-pred']
    test_orig.columns = ['price_in_USD-test']

    plt.figure()
    plt.figtext(.2,.15, 'rmse=%f' % (rmse))
    plt.plot(test_orig['price_in_USD-test'], label='test set', color='blue')
    plt.plot(pred_orig['price_in_USD-pred'], label='prediction', color='red')
    plt.title('VAR Modeling - Actual vs Predict ' + value)
    plt.xlabel('period')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('./plot-var-model/transformed/pred/'+ value + '.png')

    model_var2 = VAR(endog=full,  exog=full['avg_temp'])
    model_fit2 = model_var2.fit()

    test_df = test.stack().reset_index()
    test_df.columns = ['period', 'name', 'price']
    test_df = test_df.drop(test_df[test_df.name == 'avg_temp-test'].index)
    date_rng = pd.date_range(start=test_df.iloc[-1,0], periods=36, freq='M')

    #create a df with next 36 months as index
    new_df = pd.DataFrame(index=date_rng)
    new_df = new_df.reset_index()
    new_df.columns=['period']
    full = full.reset_index()
    new_df ['avg_temp'] = full['avg_temp'].iloc[:-36]
    full = full.set_index('period')
    new_df = new_df.set_index('period')
    cols = full.columns

    #predict next 36 months into the future
    future_val = model_fit2.forecast(model_fit2.y, steps=36,  exog_future=new_df['avg_temp'])
    future = pd.DataFrame(index=new_df.index, columns=[cols])
    for j in range(0, 2):
        for i in range(0, len(new_df)):
            future.iloc[i][j] = future_val[i][j]

    plt.figure()
    plt.figtext(.5,.20,  value)
    plt.plot(future['price_in_USD'], label='future', color='red')
    plt.plot(full ['price_in_USD'], label='past', color='blue')
    plt.title('VAR Modeling - Future ' + value)
    plt.xlabel('period')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('./plot-var-model/transformed/future/'+ value + '.png')

    future_orig = pd.DataFrame(index=future.index)
    full_orig = pd.DataFrame(index=full.index)

    future_orig['price_in_USD'] = future['price_in_USD'].astype('float32')
    full_orig['price_in_USD'] = full['price_in_USD'].astype('float32')

    #inverse-transform before plotting
    if (difftype_price =='log'):
        future_orig  = reverse_log(future_orig )
        full_orig  = reverse_log(full_orig)

    elif (difftype_price =='logb'):
        movingAvg2 =  df_log['price_in_USD'].diff(1).rolling(window=12).mean()
        full_orig['price_in_USD'] = full_orig['price_in_USD'] + movingAvg2

        #reverse future values
        future_orig = future_orig.reset_index()
        future_orig['price_in_USD'] = reverse_log(future_orig['price_in_USD'])

        #reverse past values
        full_orig = full_orig.reset_index()
        full_orig['price_in_USD'] = reverse_log(full_orig['price_in_USD'])

        future_orig = future_orig.set_index('period')
        full_orig = full_orig.set_index('period')

    elif (difftype_price == 'logD1'):
        val_future = df_log.tail(1)['price_in_USD']
        val_full = df_log.head(1)['price_in_USD']
        new_series_future = pd.Series(val_future).append(pd.Series(future_orig['price_in_USD']))
        new_series_future.columns = ['price_in_USD']
        new_series_full = pd.Series(val_full).append(pd.Series(full_orig['price_in_USD']))
        new_series_full.columns = ['price_in_USD']

        #reverse future values
        future_orig = future_orig.reset_index()
        future_orig['price_in_USD'] = undiff(new_series_future, 1)
        future_orig['price_in_USD'] = reverse_log(future_orig['price_in_USD'])

        #reverse past values
        full_orig = full_orig.reset_index()
        full_orig['price_in_USD'] = undiff(new_series_full, 1)
        full_orig['price_in_USD'] = reverse_log(full_orig['price_in_USD'])

        future_orig = future_orig.set_index('period')
        full_orig = full_orig.set_index('period')

    elif (difftype_price == 'logD2'):
        val_future = df_log.tail(2)['price_in_USD']
        val_full = df_log.head(2)['price_in_USD']
        new_series_future = pd.Series(val_future).append(pd.Series(future_orig['price_in_USD']))
        new_series_future.columns = ['price_in_USD']
        new_series_full = pd.Series(val_full).append(pd.Series(full_orig['price_in_USD']))
        new_series_full.columns = ['price_in_USD']

        #reverse future values
        future_orig = future_orig.reset_index()
        future_orig['price_in_USD'] = undiff(new_series_future, 2)
        future_orig['price_in_USD'] = reverse_log(future_orig['price_in_USD'])

        #reverse past values
        full_orig = full_orig.reset_index()
        full_orig['price_in_USD'] = undiff(new_series_full, 2)
        full_orig['price_in_USD'] = reverse_log(full_orig['price_in_USD'])

        future_orig = future_orig.set_index('period')
        full_orig = full_orig.set_index('period')

    elif (difftype_price == 'logD5'):
        val_future = df_log.tail(5)['price_in_USD']
        val_full = df_log.head(5)['price_in_USD']

        new_series_future = pd.Series(val_future).append(pd.Series(future_orig['price_in_USD']))
        new_series_future.columns = ['price_in_USD']
        new_series_full = pd.Series(val_full).append(pd.Series(full_orig['price_in_USD']))
        new_series_full.columns = ['price_in_USD']

        #reverse future values
        future_orig = future_orig.reset_index()
        future_orig['price_in_USD'] = undiff(new_series_future, 5)
        future_orig['price_in_USD'] = reverse_log(future_orig['price_in_USD'])

        #reverse past values
        full_orig = full_orig.reset_index()
        full_orig['price_in_USD'] = undiff(new_series_full, 5)
        full_orig['price_in_USD'] = reverse_log(full_orig['price_in_USD'])

        future_orig = future_orig.set_index('period')
        full_orig = full_orig.set_index('period')

    elif (difftype_price == 'logbD1'):
        val_future = df_log.tail(1)['price_in_USD']
        val_full = df_log.head(1)['price_in_USD']

        movingAvg2 =  df_log['price_in_USD'].diff(1).rolling(window=12).mean()
        full_orig['price_in_USD'] = full_orig['price_in_USD'] + movingAvg2

        new_series_future = pd.Series(val_future).append(pd.Series(future_orig['price_in_USD']))
        new_series_future.columns = ['price_in_USD']
        new_series_full = pd.Series(val_full).append(pd.Series(full_orig['price_in_USD']))
        new_series_full.columns = ['price_in_USD']

        #reverse future values
        future_orig = future_orig.reset_index()
        future_orig['price_in_USD'] = undiff(new_series_future, 1)
        future_orig['price_in_USD'] = reverse_log(future_orig['price_in_USD'])

        #reverse past values
        full_orig = full_orig.reset_index()
        full_orig['price_in_USD'] = undiff(new_series_full, 1)
        full_orig['price_in_USD'] = reverse_log(full_orig['price_in_USD'])

        future_orig = future_orig.set_index('period')
        full_orig = full_orig.set_index('period')

    elif (difftype_price == 'logbD3'):
        val_future = df_log.tail(3)['price_in_USD']
        val_full = df_log.head(3)['price_in_USD']

        movingAvg2 =  df_log['price_in_USD'].diff(3).rolling(window=12).mean()
        full_orig['price_in_USD'] = full_orig['price_in_USD'] + movingAvg2

        new_series_future = pd.Series(val_future).append(pd.Series(future_orig['price_in_USD']))
        new_series_future.columns = ['price_in_USD']
        new_series_full = pd.Series(val_full).append(pd.Series(full_orig['price_in_USD']))
        new_series_full.columns = ['price_in_USD']

        # reverse future values
        future_orig = future_orig.reset_index()
        future_orig['price_in_USD'] = undiff(new_series_future, 3)
        future_orig['price_in_USD'] = reverse_log(future_orig['price_in_USD'])

        # reverse past values
        full_orig = full_orig.reset_index()
        full_orig['price_in_USD'] = undiff(new_series_full, 3)
        full_orig['price_in_USD'] = reverse_log(full_orig['price_in_USD'])

        future_orig = future_orig.set_index('period')
        full_orig = full_orig.set_index('period')

    else:
        future_orig = future.copy()
        full_orig = full.copy()

    plt.figure()
    mean_VAR = future_orig['price_in_USD'].mean() - full_orig ['price_in_USD'].mean()
    plt.figtext(.2,.15, 'mean diff=%f' % mean_VAR)
    plt.plot(future_orig['price_in_USD'], label='future', color='red')
    plt.plot(full_orig['price_in_USD'], label='past', color='blue')
    plt.title('VAR Modeling - Future ' + value)
    plt.xlabel('period')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('./plot-var-model/transformed/future/origval/'+ value + '.png')

    return mean_VAR, rmse

def box_cox_transform(df):
    df_transform, lam = boxcox(df)
    return df_transform, lam

def lstm_model(train_X, train_y, test_X, test_y, value):
    lstm_model = Sequential()
    lstm_model.add(LSTM(120, input_shape=(train_X.shape[1], train_X.shape[2])))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mae', optimizer='adam')
    history = lstm_model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    yhat = lstm_model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    #reverse actual and predicted values
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    #plot prediction and actual data
    rmse = sqrt(mean_squared_error(test_y, yhat))
    mape = mean_absolute_percentage_error(test_y, yhat)
    pce = percentage_error_to_mean(np.mean(test_y), rmse)
    print('RMSE :' + str(rmse), file=text_file6)
    print('MAPE :' + str(mape), file=text_file6)
    plt.figure()
    plt.figtext(.2,.15, 'rmse=%f' % (rmse))
    plt.plot(inv_y, color='blue', label='test set')
    plt.plot(inv_yhat, color='red', label='prediction')
    plt.xlabel('Period')
    plt.ylabel('Food Price')
    plt.title('LSTM modeling - Actual vs Predict '+ value)
    plt.legend()
    fig = plt.gcf()
    fig.savefig('./plot-lstm-model/transformed/pred/'+ value + '.png')

    return rmse

#======================================================================================================
#start processing
#======================================================================================================

#temp
trans_list_temp =[]
trans_list_temp2 = ['Indonesia', 'Benin', 'Burkina Faso','India', 'El Salvador', 'Nepal']
#prices
trans_list_prices =[  'Central African Republic']
trans_list_prices2 = ['Guatemala' , 'Nepal' , 'Senegal']

#for VAR and LSTM
nons_list= ['El Salvador']
nons_list1=['Tajikistan', 'Bangladesh', 'Central African Republic', 'Chad', 'Costa Rica']
nons_list2=[ 'Mali', 'Senegal', 'Benin', 'Burundi', 'Niger', 'Burkina Faso'] #el salvador prev
nons_list3=['India']
nons_list4=['Mozambique']
nons_list5=[ 'Nepal']
nons_list6=[ 'Kenya']
nons_list7=['Peru' ]
nons_list8=[ 'Guatemala' ]

#create a dict for unique list of countries
df_orig = pd.read_csv('.\Market_food_prices_w_temp_reduced.csv')
country_dict = dict(zip(df_orig.country_id,df_orig.country))
list=pd.DataFrame(index=country_dict)
#initialize files
text_file1 = open("./plot-var-model/transformed/VAR_modeling_summary.doc", "w")
text_file2 = open("./plot-var-model/transformed/VAR-adf-results-avgtemp.doc", "w")
text_file3 = open("./plot-var-model/transformed/VAR-adf-results-foodprices.doc", "w")
text_file4 = open("./plot-auto-arima/transformed/Auto-ARIMA_modeling_summary-avgtemp.doc", "w")
text_file5 = open("./plot-auto-arima/transformed/Auto-ARIMA_modeling_summary-foodprices.doc", "w")
text_file6 = open("./plot-lstm-model/transformed/lstm_modeling_summary.doc", "w")
text_file7 = open("./plot-auto-arima/transformed/misc.doc", "w")
text_file8 = open("./model_summary.csv", "w")

for key, value in country_dict.items():
    df_country = pd.read_csv('.\countries\country_df_'+ value +'.csv')
    df_country.sort_values(by='period')
    df_country['period'] = pd.to_datetime(df_country.period, format='%Y-%m-%d')
    drop_cols = ['country_id', 'country', 'commodity_purchased', 'commodity_purchase_id']
    df_country.drop(drop_cols, axis=1, inplace=True)
    df_country_merge = pd.DataFrame()
    df_country_merge = df_country.groupby(['period','avg_temp'])['price_in_USD'].mean().reset_index()
    list.at[key , 'country'] = value

    print(str(value), file=text_file7)
    print(df_country_merge.head(20), file = text_file7)

    df_orig1 = df_country_merge.copy()

    print('='*70, file=text_file1)
    print(str(value), file=text_file1)
    print('=' * 70, file=text_file1)
    print(str(value))
    #
    #avg temp
    df_temp = df_country.groupby('period')['avg_temp'].mean().reset_index()
    df_temp.set_index('period', inplace=True)
    df_temp = df_temp.sort_index()
    df_temp_full = df_temp.copy()
    difftype='no_trans'
    lam=0
    df_train_1, df_test_1 = split_data(df_temp, 'Y')
    df_transform_temp = df_train_1.copy()
    df_test_temp = df_test_1.copy()
    if value in trans_list_temp:#['Benin', 'Burkina Faso','India', 'El Salvador', 'Nepal']
        df_transform_temp ['avg_temp'], lam = box_cox_transform(df_train_1 ['avg_temp'])
        df_temp_full ['avg_temp'], lam = box_cox_transform(df_temp ['avg_temp'])
        df_test_temp['avg_temp'], lam = box_cox_transform(df_test_1['avg_temp'])
        difftype = 'bc'
    elif value in trans_list_temp2: #Indonesia
        df_transform_temp['avg_temp'] = np.log(df_train_1 ['avg_temp'])
        df_temp_full ['avg_temp'] = np.log(df_temp['avg_temp'])
        df_test_temp['avg_temp'] = np.log(df_test_1['avg_temp'])
        difftype = 'log'
    #call auto arima modeling
    mean_diff, rmse_temp_ARIMA = auto_arima_model(df_transform_temp, df_test_temp, value, 'avgtemp', text_file4, True, df_temp_full, difftype, lam)

    #food prices
    df_prices = df_country.groupby('period')['price_in_USD'].mean().reset_index()
    df_prices.set_index('period', inplace=True)
    df_prices = df_prices.sort_index()
    df_prices_full = df_prices.copy()
    df_train_2, df_test_2 = split_data(df_prices, 'Y')
    df_transform_prices = df_train_2.copy()
    df_test_prices = df_test_2.copy()
    if value in trans_list_prices:
        df_transform_prices ['price_in_USD'], lam = box_cox_transform(df_train_2['price_in_USD'])
        df_prices_full ['price_in_USD'], lam = box_cox_transform(df_prices ['price_in_USD'])
        df_test_prices['price_in_USD'], lam = box_cox_transform(df_test_2['price_in_USD'])
        difftype = 'bc'
    elif value in trans_list_prices2:
        df_transform_prices['price_in_USD'] = np.log(df_train_2['price_in_USD'])
        df_prices_full['price_in_USD']  = np.log(df_prices['price_in_USD'])
        df_test_prices['price_in_USD'] = np.log(df_test_2['price_in_USD'])
        difftype = 'log'
    mean_diff, rmse_prices_ARIMA = auto_arima_model(df_transform_prices, df_test_prices, value, 'foodprices', text_file5, False, df_prices_full, difftype, lam)
    list.at[key , 'mean_diff_ARIMA_prices'] = mean_diff
    list.at[key, 'rmse_prices_ARIMA'] = rmse_prices_ARIMA

    #making the series stationary for VAR
    df_full_var = df_country_merge.copy()
    df_train_3, df_test_3 = split_data(df_country_merge, 'Y')
    if value in nons_list:
        df_train_var, log = make_stationary(df_train_3, value, 'T0', 'train')
        df_test_var, log = make_stationary(df_test_3, value, 'T0', '')
        df_full_var, log =  make_stationary(df_country_merge, value, 'T0', '')
        difftype_price = 'logb'
    elif value in nons_list1:
        df_train_var, log = make_stationary(df_train_3, value, 'T1' , 'train')
        df_test_var, log = make_stationary(df_test_3, value, 'T1', '')
        df_full_var, log = make_stationary(df_country_merge, value, 'T1', '')
        difftype_price = 'logD1'
    elif value in nons_list2:
        df_train_var, log = make_stationary(df_train_3, value, 'T2', 'train')
        df_test_var, log = make_stationary(df_test_3, value, 'T2', '')
        df_full_var, log = make_stationary(df_country_merge, value, 'T2', '')
        difftype_price = 'logbD1'
    elif value in nons_list3:
        df_train_var, log = make_stationary(df_train_3, value, 'T3', 'train')
        df_test_var, log = make_stationary(df_test_3, value, 'T3', '')
        df_full_var, log = make_stationary(df_country_merge, value, 'T3', '')
        difftype_price = 'logbD3'
    elif value in nons_list4:
        df_train_var, log = make_stationary(df_train_3, value, 'T4', 'train')
        df_test_var, log = make_stationary(df_test_3, value, 'T4', '')
        df_full_var, log = make_stationary(df_country_merge, value, 'T4', '')
        difftype_price = 'logD5'
    elif value in nons_list5:
        df_train_var, log = make_stationary(df_train_3, value, 'T5', 'train')
        df_test_var, log = make_stationary(df_test_3, value, 'T5', '')
        df_full_var, log = make_stationary(df_country_merge, value, 'T5', '')
        difftype_price = 'logbD1'
    elif value in nons_list6:
        df_train_var, log = make_stationary(df_train_3, value, 'T6' ,'train')
        df_test_var, log = make_stationary(df_test_3, value, 'T6', '')
        df_full_var, log = make_stationary(df_country_merge, value, 'T6', '')
        difftype_price = 'logbD1'
    elif value in nons_list7:
        df_train_var, log = make_stationary(df_train_3, value, 'T7','train')
        df_test_var, log = make_stationary(df_test_3, value, 'T7', '')
        df_full_var, log = make_stationary(df_country_merge, value, 'T7', '')
        difftype_price = 'logbD1'
    elif value in nons_list8:
        df_train_var, log = make_stationary(df_train_3, value, 'T8','train')
        df_test_var, log = make_stationary(df_test_3, value, 'T8', '')
        df_full_var, log = make_stationary(df_country_merge, value, 'T8', '')
        difftype_price = 'logD1'
    else:
        df_train_var, log = make_stationary(df_train_3, value, 'N', 'train')
        df_test_var, log = make_stationary(df_test_3, value, 'N', '')
        df_full_var, log = make_stationary(df_country_merge, value, 'N', '')
        difftype_price = 'no_trans'

    #call VAR modeling
    mean_diff, rmse_VAR = var_model(df_train_var, df_test_var, value, df_full_var, difftype_price, log, df_train_3, df_test_3)
    list.at[key, 'mean_diff_VAR'] = mean_diff
    list.at[key, 'rmse_VAR'] = rmse_VAR

    values = df_country_merge.values
    values = values.astype('float32')
    df_train_4, df_test_4 = split_data(values, 'N')
    #train set
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df_train_4)
    reframed = series_to_supervise(scaled, 1, 1)
    reframed.drop(reframed.columns[[0]], axis=1, inplace=True)
    df_train_4 = reframed.values
    #test set
    scaled = scaler.fit_transform(df_test_4)
    reframed = series_to_supervise(scaled, 1, 1)
    reframed.drop(reframed.columns[[0]], axis=1, inplace=True)
    df_test_4 = reframed.values

    train_X, train_y = df_train_4[:, :-1], df_train_4[:, -1]
    test_X, test_y = df_test_4[:, :-1], df_test_4[:, -1]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    rmse_LSTM  = lstm_model(train_X, train_y, test_X, test_y, value)

    list.at[key, 'rmse_LSTM'] = rmse_LSTM
    list.at[key, 'dataset size'] = len(df_orig1)

#write foreasting summary
list.to_csv(text_file8, index=False)