import json
import pandas as pd

#Convert UOM to the base unit
#create a data dictionary for the UOM conversion
uom_map_day = {
    'day': 1,
    'month': 30
}

uom_map_mass = {
    'kg': 1,
    'gm': 0.001,
    'mt': 1000,
    'lb': 0.453592,
    'mg': 0.000001,
    'g': 0.001,
    'pound': 0.453592,
    'sack': 165.10762,
    'bunch': 0.34,
    'cubic': 2406.53,
    'libra': 0.323,
    'packet': 0.014,
    'loaf': 0.31,
    'head': 0.09997,
    'day':  1016.04608,
    'tubers': 2.5
}

uom_map_pcs = {
    'pcs': 1,
    'dozen': 12,
    'unit' : 1,
    'loaf': 1,
    'marmite':1
}

uom_map_volume = {
    'l': 1,
    'gallon': 3.78541,
    'ml': 0.001,
    'cuartilla': 13.88
}

global_food_prices = pd.read_csv('.\Market_food_prices_w_temp_USD.csv')

def convert_unit(row):
    quantity, uom, price_in_USD = row.measurement_const, row.measurement_unit_type.lower(),row.price_in_USD
    if uom in uom_map_day:
        multiplier = uom_map_day[uom]
        return [quantity * multiplier, 'day', price_in_USD/(quantity * multiplier)]
    elif uom in uom_map_mass:
        multiplier = uom_map_mass[uom]
        return [quantity * multiplier, 'kg', price_in_USD/(quantity * multiplier)]
    elif uom in uom_map_pcs:
        multiplier = uom_map_pcs[uom]
        return [quantity * multiplier, 'pcs' , price_in_USD/(quantity * multiplier)]
    else:
        multiplier = uom_map_volume[uom]
        return [quantity * multiplier, 'l', price_in_USD/(quantity * multiplier)]


#convert to base unit - mass(KG), volume (liters), pcs
global_food_prices[['converted_const', 'converted_unit', 'price_in_USD']] = global_food_prices.apply(lambda row: pd.Series(convert_unit(row)), axis=1)

#write dataframe to csv file
global_food_prices.to_csv('.\Market_food_prices_w_temp_converted.csv', index=False)

