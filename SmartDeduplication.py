import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

restaurant_db_cols = ['Name', 'Address', 'Phone', 'Style']
restaurant_db = pd.read_csv(r'RestaurantListComma.csv', sep=',', names=restaurant_db_cols, encoding="latin-1")
pd.set_option("display.max_rows", None, "display.max_columns", None)

restaurant_db['Name'] = restaurant_db['Name'].map(lambda x: re.sub(r'\W+', '', x))
restaurant_db['Phone'] = restaurant_db['Phone'].map(lambda x: re.sub(r'\W+', '', x))


def convert_frame_to_array(restaurant_db_in):
    restaurant_db_array_out = restaurant_db_in.to_numpy()
    return restaurant_db_array_out

"""
    Fucntion to convert the hash value for each column.
    We are using the Horners method that is used in the Rabin Karp method for creating the hash value.
    This function returns a dataframe that has an additional column "data_hash" which is the total sum of hash values of each column.
"""


def create_hash_column(restaurant_db_array_in):
    power_value = 256
    mod_value = 100003
    data_hash = []

    for row in restaurant_db_array_in:
        total_sum_column = 0
        for column in row:
            value_length = len(column)
            hash_value = 0
            column = column.upper()
            for char in range(value_length):
                #print(column[char])
                hash_value = (hash_value * power_value + ord(column[char])) % mod_value
                total_sum_column += hash_value
        data_hash.append(total_sum_column)
    restaurant_db['DataHash'] = data_hash
    return restaurant_db

#print(type(restaurantDb['data_hash']))

# Driver code to call the necessary functions


restaurant_db_array = convert_frame_to_array(restaurant_db)
create_hash_column(restaurant_db_array)
print(restaurant_db)