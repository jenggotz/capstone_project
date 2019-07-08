import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries
import matplotlib as mpl
import seaborn as sns

#using the final file with only 19 countries
merged_file = pd.read_csv('.\Market_food_prices_w_temp_reduced.csv')
merged_file = merged_file.sort_values('period')
df = merged_file.groupby('country')
idx = df.nth(1).index
plt.rcParams["figure.figsize"] = [16, 12]

#timeseries plot based on avg temp
print('plotting timeseries avg temp...')
for i in range(len(idx.tolist())):
    df = merged_file[(merged_file['country'] == idx.tolist()[i])]
    df['period'] = pd.to_datetime(df.period, format='%Y-%m-%d')
    df = df.groupby('period')['avg_temp'].mean().reset_index()
    df.set_index('period', inplace=True)
    df['avg_temp'].resample('M').apply([np.mean]).plot()
    df.plot()
    plt.title(idx.tolist()[i])
    if len(df)%2 == 0:
        split = int(len(df) / 2)
    else:
        split = int((len(df) + 1)/2)

    X1, X2 = df[0:split], df[split:]
    mean1, mean2 = X1.mean(), X2.mean()
    var1, var2 = X1.var(), X2.var()

    print(idx.tolist()[i])
    print('mean1=%f, mean2=%f' % (mean1, mean2))
    print('variance1=%f, variance2=%f' % (var1, var2))
    plt.xticks(rotation=45)
    plt.figtext(.2,.15, 'mean1=%f, mean2=%f' % (mean1, mean2))
    plt.figtext(.2,.1, 'variance1=%f, variance2=%f' % (var1, var2))
    fig = plt.gcf()
    fig.savefig('./plot-timeseries-avgtemp/reduced/'+ idx.tolist()[i] + '.png')

#timeseries plot based on food prices
print('plotting timeseries food prices...')
for i in range(len(idx.tolist())):
    df = merged_file[(merged_file['country'] == idx.tolist()[i])]
    df['period'] = pd.to_datetime(df.period, format='%Y-%m-%d')
    df = df.groupby('period')['price_in_USD'].mean().reset_index()
    df.set_index('period', inplace=True)
    df['price_in_USD'].resample('M').apply([np.mean]).plot()
    df.plot()
    plt.title(idx.tolist()[i])
    if len(df)%2 == 0:
        split = int(len(df) / 2)
    else:
        split = int((len(df) + 1)/2)
    X1, X2 = df[0:split], df[split:]
    mean1, mean2 = X1.mean(), X2.mean()
    var1, var2 = X1.var(), X2.var()
    print(idx.tolist()[i])
    print('mean1=%f, mean2=%f' % (mean1, mean2))
    print('variance1=%f, variance2=%f' % (var1, var2))
    plt.xticks(rotation=45)
    plt.figtext(.2,.15, 'mean1=%f, mean2=%f' % (mean1, mean2))
    plt.figtext(.2,.1, 'variance1=%f, variance2=%f' % (var1, var2))
    fig = plt.gcf()
    fig.savefig('./plot-timeseries-foodprices/reduced/'+ idx.tolist()[i] + '.png')

#seasonal plot of time series (food prices)
print('plotting seasonal plots of time series - food prices...')
for i in range(len(idx.tolist())):
    df = merged_file[(merged_file['country'] == idx.tolist()[i])]
    df = df.groupby('period')['price_in_USD'].mean().reset_index()
    #prepare data
    df['period'] = pd.to_datetime(df.period, format='%Y-%m-%d')
    df['year'] = [d.year for d in df.period]
    df['month'] = [d.strftime('%b') for d in df.period]
    years = df['year'].unique()

    #prepare colors
    np.random.seed(100)
    mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

    #draw plot
    plt.figure(dpi= 80)
    for w, y in enumerate(years):
        if w > 0:
            plt.plot('month', 'price_in_USD', data=df.loc[df.year==y, :], color=mycolors[w], label=y)
            plt.text(df.loc[df.year==y, :].shape[0]-.9, df.loc[df.year==y, 'price_in_USD'][-1:].values[0], y, fontsize=12, color=mycolors[w])

    plt.title("Seasonal Plot of Food Prices Time Series - " + idx.tolist()[i] , fontsize=20)
    fig = plt.gcf()
    fig.savefig('./seasonal_plot/food_prices/'+ idx.tolist()[i] + '.png')

#seasonal plot of time series (food prices)
print('plotting seasonal plots of time series - avg temp...')
for i in range(len(idx.tolist())):
    df = merged_file[(merged_file['country'] == idx.tolist()[i])]
    df = df.groupby('period')['avg_temp'].mean().reset_index()
    #prepare data
    df['period'] = pd.to_datetime(df.period, format='%Y-%m-%d')
    df['year'] = [d.year for d in df.period]
    df['month'] = [d.strftime('%b') for d in df.period]
    years = df['year'].unique()

    #prepare colors
    np.random.seed(100)
    mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

    #draw plot
    plt.figure(dpi= 80)
    for w, y in enumerate(years):
        if w > 0:
            plt.plot('month', 'avg_temp', data=df.loc[df.year==y, :], color=mycolors[w], label=y)
            plt.text(df.loc[df.year==y, :].shape[0]-.9, df.loc[df.year==y, 'avg_temp'][-1:].values[0], y, fontsize=12, color=mycolors[w])

    plt.title("Seasonal Plot of Avg Temp Time Series - "+ idx.tolist()[i], fontsize=20)
    fig = plt.gcf()
    fig.savefig('./seasonal_plot/avg_temp/'+ idx.tolist()[i] + '.png')

#plot scatter plot for relationship between temp and food prices
print('plotting scatterplot...')
for i in range(len(idx.tolist())):
    plt.figure()
    df = merged_file[(merged_file['country'] == idx.tolist()[i])]
    df['period'] = pd.to_datetime(df.period, format='%Y-%m-%d')
    df = df.groupby(['period', 'avg_temp', 'commodity_purchased'])['price_in_USD'].mean().reset_index()
    df.set_index('period', inplace=True)
    sns.scatterplot(x='avg_temp', y='price_in_USD', data=df, hue='commodity_purchased')
    plt.xticks(rotation=45)
    plt.title("Scatterplot - relationship of temp and prices - " + idx.tolist()[i], fontsize=10, )
    fig = plt.gcf()
    fig.savefig('./plot-scatterplot/'+ idx.tolist()[i] + '.png')

#boxplots for avg temp and food prices in 1 plot
print('plotting boxplot...')
for i in range(len(idx.tolist())):
    plt.figure()
    df = merged_file[(merged_file['country'] == idx.tolist()[i])]
    df['period'] = pd.to_datetime(df.period, format='%Y-%m-%d')
    df['year'] = [d.year for d in df.period]
    df['month'] = [d.strftime('%b') for d in df.period]

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    for name, ax in zip(['avg_temp', 'price_in_USD'], axes):
        sns.boxplot(data=df, x='month', y=name, ax=ax)
        ax.set_title(name)

    fig = plt.gcf()
    fig.savefig('./plot-all-variables/boxplot/'+ idx.tolist()[i] + '.png')