import urllib.request
import json
import pandas as pd

#Convert currency to USD
class currencyConverter:
    rates = {}
    def __init__(self, url):
        req = urllib.request.Request(url, headers={'User-Agent':'currency converter'})
        data = urllib.request.urlopen(req).read()
        data = json.loads(data.decode('utf-8'))
        self.rates = data['rates']

    def convert(self, amt, from_currency, to_currency):
        init_amt = amt
        if from_currency != 'EUR':
            amt = amt / self.rates[from_currency]
        if to_currency == 'EUR':
            return amt
        else:
            return amt * self.rates[to_currency]

#read input file
global_food_prices = pd.read_csv('.\Market_food_prices_w_temp.csv')

#change currency to the abbreviated value
global_food_prices.loc[global_food_prices['currency_name'].str.contains('Somaliland Shilling'), 'currency_name'] = 'SOS'

#call currency converter API to convert all prices to USD
curr_converter = currencyConverter('http://data.fixer.io/api/latest?access_key=5347d95a706d89fbb2690f907d1fe618')
global_food_prices['price_in_USD'] = global_food_prices.apply(lambda row: curr_converter.convert(row['price_paid'], row['currency_name'], 'USD'), axis=1)

#write dataframe to csv
global_food_prices.to_csv('.\Market_food_prices_w_temp_USD.csv', index=False)

