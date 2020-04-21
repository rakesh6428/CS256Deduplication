import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

restaurantDb_cols = ['Name', 'Address', 'Phone', 'Style']
restaurantDb = pd.read_csv(r'RestaurantListComma.csv', sep=',', names=restaurantDb_cols, encoding="latin-1")
pd.set_option("display.max_rows", None, "display.max_columns", None)


restaurantDb['Name'] = restaurantDb['Name'].map(lambda x: re.sub(r'\W+', '', x))
restaurantDb['Phone'] = restaurantDb['Phone'].map(lambda x: re.sub(r'\W+', '', x))

print(restaurantDb.head())

