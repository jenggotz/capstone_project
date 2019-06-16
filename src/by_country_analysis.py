import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries
import os

#file comes from jupyter by country initial analysis
merged_file = pd.read_csv('.\Market_food_prices_w_temp_grouped.csv')
merged_file = merged_file.sort_values('period')
df = merged_file.groupby('country')
idx = df.nth(1).index

if os.path.exists('.\outliers-numeric_cols_country.csv'):
   os.remove('.\outliers-numeric_cols_country.csv')

if os.path.exists('.\commodity_count.csv'):
   os.remove('.\commodity_count.csv')

#check the distribution of categorical variables
#draw boxplot for avg temp per country
for i in range(len(idx.tolist())):
    df = pd.DataFrame()
    df = merged_file[(merged_file['country'] == idx.tolist()[i])]
    df.set_index('period')
    print("="*90)
    print(idx.tolist()[i] +  '\t' + 'IQR')
    print("="*90)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    print (IQR)
    #check if there's an outlier for each country and write to a file
    df_num = df[['avg_temp', 'price_in_USD']]
    df_result = (df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR))
    df_join = pd.concat([df, df_result], axis=1)
    df_join.set_index('period')
    with open('.\outliers-numeric_cols_country.csv', 'a', newline='') as f:
        df_join.to_csv(f, mode='a', index=None,  header=f.tell()==0)
    #initialize the plot
    fig = plt.figure()
    plt.title(idx.tolist()[i])
    sns.boxplot(x=df['avg_temp'], width=0.8, palette="colorblind")
    sns.swarmplot(x=df['avg_temp'], color=".25")
    fig = plt.gcf()
    figsize = fig.get_size_inches()
    fig.set_size_inches(figsize * 2.5)
    fig.savefig('./plot-boxplot-avgtemp/' + idx.tolist()[i] + '.png')

print('plotting boxplots for commodity...')
#draw boxplot per country for prices
for i in range(len(idx.tolist())):
    df = pd.DataFrame()
    df = merged_file[(merged_file['country'] == idx.tolist()[i])]
    df.set_index('period')
    # initialize the plot
    fig = plt.figure()
    plt.title(idx.tolist()[i])
    sns.boxplot(x=df['commodity_purchased'],y=df['price_in_USD'], width=0.8, palette="colorblind")
    sns.swarmplot(x=df['commodity_purchased'],y=df['price_in_USD'], color=".25")
    fig = plt.gcf()
    figsize = fig.get_size_inches()
    fig.set_size_inches(figsize * 2.5)
    fig.savefig('./plot-boxplot-commodity/' + idx.tolist()[i] + '.png')

print('plotting histogram per country...')
#plot distribution of numeric variables per country
for i in range(len(idx.tolist())):
    df = pd.DataFrame()
    df = merged_file[(merged_file['country'] == idx.tolist()[i])]
    df.set_index('period')
    df.hist()
    fig = plt.gcf()
    fig.text(0.5, 0.04, 'values', ha='center')
    fig.text(0.04, 0.5, 'count', va='center', rotation='vertical')
    figsize = fig.get_size_inches()
    fig.set_size_inches(figsize * 2.5)
    fig.savefig('./plot-hist-country/' + idx.tolist()[i] + '.png')

print('pivot on country and commodity...')
#pivot by commodity and country
df_pivot1 = pd.DataFrame(pd.pivot_table(merged_file,index=['country','commodity_purchased'], values=['period'],aggfunc=[len,np.min,np.max]).reset_index())
with open('.\pivot_country_commodity.csv', 'w', newline='') as f:
    df_pivot1.to_csv(f, index=None)
df_pivot1.reset_index()
df_pivot1.columns = ['country', 'commodity_purchased', 'count_period', 'min_period', 'max_period']

print('plotting barplots for commodity...')
#plot how many rows of data do we have per commodity
for i in range(len(idx.tolist())):
    df = df_pivot1[(df_pivot1['country'] == idx.tolist()[i])]
    #initialize the plot
    fig = plt.figure()
    plt.title(idx.tolist()[i])
    x = sns.barplot(x=df['commodity_purchased'],y=df['count_period'],palette="colorblind")
    fig = plt.gcf()
    figsize = fig.get_size_inches()
    fig.set_size_inches(figsize * 3.5)
    fig.savefig('./plot-countplot-commodity/' + idx.tolist()[i] + '.png')

print('pivot on period and commodity...')
#pivot by commodity and period
df_pivot2 = pd.pivot_table(merged_file,index=['period','commodity_purchased'], values=['country'],aggfunc=len).reset_index()
with open('.\pivot_period_commodity.csv', 'w', newline='') as f:
    df_pivot2.to_csv(f, index=None)
df_pivot2.reset_index()
df_pivot2.columns = ['period', 'commodity_purchased', 'count']

print('printing commodity count...')
#count the no. of months per commodity in each country
for i in range(len(idx.tolist())):
    df = merged_file[(merged_file['country'] == idx.tolist()[i])]
    df_commodity = df[['commodity_purchased', 'period']]
    df_commodity.set_index('period')
    df_group = df_commodity.groupby(['commodity_purchased'])
    with open('.\commodity_count.csv', 'a', newline='') as f:
        print(idx.tolist()[i], file=f)
        df_group.count().reset_index()[['commodity_purchased', 'period']].to_csv(f, mode='a', index=None,  header=f.tell()==0)

print('plot count of months per commodity...')
#plot how many months of data do we have per commodity
for i in range(len(idx.tolist())):
    df = merged_file[(merged_file['country'] == idx.tolist()[i])]
    #initialize the plot
    df['year'] = pd.DatetimeIndex(df['period']).year
    fig = plt.figure()
    ax=(df
     .groupby(['year', 'commodity_purchased'])
     .size()
     .unstack()
     .plot.bar()
     )
    plt.title(idx.tolist()[i])
    ax.set_xlabel("year")
    ax.set_ylabel("no. of months")
    fig = plt.gcf()
    figsize = fig.get_size_inches()
    fig.set_size_inches(figsize * 3.5)
    fig.savefig('./plot-countmonths-commodity/' + idx.tolist()[i] + '.png')
