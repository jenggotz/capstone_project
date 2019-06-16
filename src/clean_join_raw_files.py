import pandas as pd

#read the temperature by city csv file
global_temp = pd.read_csv('.\GlobalLandTemperaturesByCountry.csv')
global_temp.columns = ['full_date', 'avg_temp', 'avg_temp_uncty','country']

#extract month and year from full date to be used to map with the 2nd file
global_temp['full_date'] = pd.to_datetime(global_temp['full_date'])
global_temp['year'], global_temp['month'] = global_temp['full_date'].dt.year, global_temp['full_date'].dt.month

#read the food prices by city csv file
global_food_prices = pd.read_csv('.\wfp_market_food_prices.csv', sep=',', encoding='iso-8859-1')
global_food_prices.columns = ['country_id',	'country', 'locality_id', 'locality_name', 'mkt_id', 'mkt_name', 'commodity_purchase_id',
                              'commodity_purchased', 'currency_id', 'currency_name', 'mkt_type_id', 'mkt_type',
                              'measurement_id',	'measurement_unit',	'month', 'year', 'price_paid', 'commodity_source']

#Extract the digit constant from the UOM column for some rows e.g. 15 KG - extract 15; KG - update to 1
global_food_prices['measurement_const'] = global_food_prices['measurement_unit'].str.extract('(\d*\.?\d+)', expand=True)
global_food_prices['measurement_const'].fillna(1,inplace=True)

#Extract the unit from the UOM column for some rows e.g. 15 KG - extract KG
global_food_prices['measurement_unit_type'] = global_food_prices['measurement_unit'].str.extract('([a-zA-Z/]+)', expand=True)

#Remove characters inside () to remove variations in the commodity
global_food_prices['commodity_purchased'] = global_food_prices['commodity_purchased'].str.replace(r'\([a-zA-Z \,-\\]+\)', '')
global_food_prices['commodity_purchased'] = global_food_prices['commodity_purchased'].str.strip()
global_food_prices['commodity_purchased'] = global_food_prices['commodity_purchased'].str.replace(' ','')
global_food_prices['commodity_source'] = global_food_prices['commodity_source'].str.strip()
global_food_prices['commodity_source'] = global_food_prices['commodity_source'].str.replace('WFP/','WFP')

#write files to csv
global_food_prices.to_csv('.\wfp_market_food_prices_filtered.csv', index=False)

#replace commodity with encoded id
labels = global_food_prices['commodity_purchased'].astype('category').cat.categories.tolist()
replace_map_comp = {'commodity_purchased' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
global_food_prices['commodity'] = global_food_prices['commodity_purchased']
global_food_prices.replace(replace_map_comp, inplace=True)
global_food_prices.drop('commodity_purchase_id', axis=1, inplace=True)
global_food_prices.rename({"commodity_purchased": "commodity_purchase_id", "commodity": "commodity_purchased"}, axis=1, inplace=True)
global_food_prices.to_csv('.\Market_food_prices_w_temp_replaced.csv', index=False)

#merge global_food_prices with temp dataframe and create a merged dataframe
merged_left = pd.merge(global_food_prices, global_temp, how='left', on=['month', 'year','country'])

#create a new field for the month and year and name as period
merged_left['period'] = pd.to_datetime(merged_left.year.astype(str) + '-' + merged_left.month.astype(str))

#drop the other fields for dates to avoid confusion when doing further analysis
cols=['full_date', 'month', 'year']
merged_left.drop(cols ,axis=1, inplace=True)

#output rows from global food prices that are not matched
no_temp_rows = merged_left[(merged_left['avg_temp'].isnull())]
no_temp_rows.to_csv('.\Market_food_prices_no_temp.csv', index=False)

#output rows from global food prices that are matched
w_temp_rows = merged_left[(merged_left['avg_temp'].notnull())]
w_temp_rows.to_csv('.\Market_food_prices_w_temp.csv', index=False)



