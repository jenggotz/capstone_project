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
import scipy.stats as ss
from numpy.random import seed
from scipy.stats import shapiro

if os.path.exists('.\merged_file-correlation-stats.csv'):
   os.remove('.\merged_file-correlation-stats.csv')

merged_file = pd.read_csv('.\Market_food_prices_w_temp_converted.csv')
merged_file = merged_file.sort_values('period')

cols = ['locality_id','locality_name','mkt_id','mkt_name','currency_id', 'currency_name',
        'measurement_id','measurement_unit','price_paid','measurement_const','measurement_unit_type','converted_const']
merged_file.drop(cols ,axis=1, inplace=True)

labels = merged_file['commodity_source'].astype('category').cat.categories.tolist()
replace_map_comp = {'commodity_source' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
merged_file_replace = merged_file.copy()
merged_file_replace.replace(replace_map_comp, inplace=True)

fig = plt.figure()
def heatMap(df, var):
    corr = df.corr(method=var)
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(0, 255, sep=8, n=200),
            square=True,
            annot=True
        )
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns, rotation=45)
    fig = plt.gcf()
    fig.savefig('./plot-initialanalysis/merged_file-'+var+'.png')

heatMap(merged_file_replace, 'pearson')
heatMap(merged_file_replace, 'spearman')
