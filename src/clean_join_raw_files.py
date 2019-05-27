import pandas as pd

#Step 1
#read the temperature by city csv file
global_temp = pd.read_csv('.\GlobalLandTemperaturesByCountry.csv')
global_temp.columns = ['full_date', 'avg_temp', 'avg_temp_uncty','country']

#extract only years from 1992 and 2017 and create a csv with the filtered rows
global_temp['full_date'] = pd.to_datetime(global_temp['full_date'])
global_temp['year'], global_temp['month'] = global_temp['full_date'].dt.year, global_temp['full_date'].dt.month
global_temp2 = global_temp[(global_temp['full_date'].dt.year >= 1992) & (global_temp['full_date'].dt.year <= 2017)]

#remove the word 'city' in the file
global_temp_obj = global_temp2.select_dtypes(['object'])
global_temp2[global_temp_obj.columns] = global_temp_obj.apply(lambda x: x.str.strip())

#read the food prices by city csv file
global_food_prices = pd.read_csv('.\wfp_market_food_prices.csv', sep=',', encoding='iso-8859-1')
global_food_prices.columns = ['country_id',	'country', 'locality_id', 'locality_name', 'market_id', 'city', 'commodity_purchase_id',
                              'commodity_purchased', 'currency_id', 'currency_name', 'market_type_id', 'market_type',
                              'measurement_id',	'measurement_unit',	'month', 'year', 'price_paid', 'commodity_source']

global_food_obj = global_food_prices.select_dtypes(['object'])
global_food_prices[global_food_obj.columns] = global_food_obj.apply(lambda x: x.str.strip())

#Extract the digit constant from the UOM column for some rows e.g. 15 KG - extract 15; KG - update to 1
global_food_prices['measurement_const'] = global_food_prices['measurement_unit'].str.extract('(\d*\.?\d+)', expand=True)
global_food_prices['measurement_const'].fillna(1,inplace=True)

#Extract the unit from the UOM column for some rows e.g. 15 KG - extract KG
global_food_prices['measurement_unit_type'] = global_food_prices['measurement_unit'].str.extract('([a-zA-Z/]+)', expand=True)

#Remove characters inside () to remove variations in the commodity
global_food_prices['commodity_purchased'] = global_food_prices['commodity_purchased'].str.replace(r'\([a-zA-Z \,-\\]+\)', '')
global_food_prices['commodity_purchased'] = global_food_prices['commodity_purchased'].str.strip()
global_food_prices['commodity_purchased'] = global_food_prices['commodity_purchased'].str.replace(' ','')

#write files to csv
global_food_prices.to_csv('.\wfp_market_food_prices_filtered.csv', index=False)

#merge global_food_prices with temp dataframe and create a merged dataframe
merged_left = pd.merge(global_food_prices, global_temp2, how='left', on=['month', 'year','country'])

#output rows from global food prices that are not matched
no_temp_rows = merged_left[(merged_left['avg_temp'].isnull())]
no_temp_rows.to_csv('.\Market_food_prices_no_temp.csv', index=False)

#output rows from global food prices that are matched
w_temp_rows = merged_left[(merged_left['avg_temp'].notnull())]
w_temp_rows.to_csv('.\Market_food_prices_w_temp.csv', index=False)

