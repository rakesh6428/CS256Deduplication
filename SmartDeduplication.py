import pandas as pd
import re
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from collections import Counter, defaultdict

restaurant_db_cols = ['Name', 'Address', 'Phone', 'Style']
restaurant_db = pd.read_csv(r'CompleteRestaurantList.csv', sep=',', names=restaurant_db_cols, encoding="latin-1")
pd.set_option("display.max_rows", None, "display.max_columns", None)


restaurant_db['Name'] = restaurant_db['Name'].map(lambda x: re.sub(r'\W+', '', x))
restaurant_db['Phone'] = restaurant_db['Phone'].map(lambda x: re.sub(r'\W+', '', x))
restaurant_db['Style'] = restaurant_db['Style'].map(lambda x: re.sub(r'\W+', '', x))



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
    mod_value = 283  #100003
    name_hash = []
    address_hash = []
    phone_hash = []
    style_hash = []
    data_hash = []

    for row in restaurant_db_array_in:
        total_sum_column = 0
        count = 1
        for column in row:
            value_length = len(column)
            hash_value = 0
            column_hash = 0
            column = column.upper()
            for char in range(value_length):

                hash_value = (hash_value * power_value + ord(column[char])) % mod_value
                column_hash = column_hash+hash_value
                total_sum_column += hash_value

            if count == 1:
                name_hash.append(column_hash)
            elif count == 2:
                address_hash.append(column_hash)
            elif count == 3:
                phone_hash.append(column_hash)
            elif count == 4:
                style_hash.append(column_hash)
            count+=1
        data_hash.append(total_sum_column)
    restaurant_db['NameHash'] = name_hash
    restaurant_db['AddressHash'] = address_hash
    restaurant_db['PhoneHash'] = phone_hash
    restaurant_db['StyleHash'] = style_hash
    restaurant_db['DataHash'] = data_hash
    return restaurant_db


def normalise_data(restaurant_db_norm, restaurant_db_normalised_df):
    for feature in restaurant_db_norm.columns:
        max_value = restaurant_db_graph[feature].max()
        min_value = restaurant_db_graph[feature].min()
        restaurant_db_normalised_df[feature] = (restaurant_db_graph[feature]-min_value)/(max_value-min_value)
    return restaurant_db_normalised_df


def find_number_of_cluster():
    distorsions = []
    for k in range(2, 20):
        kmeans_cluster = KMeans(n_clusters=k)
        kmeans_cluster.fit(restaurant_db_graph_normalised)
        distorsions.append(kmeans_cluster.inertia_)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, 20), distorsions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.show()


# Driver code to call the necessary functions


restaurant_db_array = convert_frame_to_array(restaurant_db)
create_hash_column(restaurant_db_array)

restaurant_db_graph = restaurant_db[['NameHash', 'AddressHash', 'PhoneHash', 'StyleHash']].copy()
restaurant_db_graph_normalised = restaurant_db_graph.copy()
restaurant_db_graph_normalised = normalise_data(restaurant_db_graph, restaurant_db_graph_normalised)
print(restaurant_db_graph_normalised)
#find_number_of_cluster()



#Running the Testing file.

Name = 'Arnie Mortons of Chicago'#input("Enter the Name of the restaurant: ")
Address = '435S.LaCienegaBlvd.LosAngeles'#input("Enter the address : ")
Phone = '310-246-1501'#input("Enter the phone number: ")
Style = 'Italian'#input("Enter the style: ")
new_restaurant = pd.DataFrame({"Name":[Name], "Address": [Address], "Phone":[Phone],"Style":[Style]})
new_restaurant['Name'] = new_restaurant['Name'].map(lambda x: re.sub(r'\W+', '', x))
new_restaurant['Phone'] = new_restaurant['Phone'].map(lambda x: re.sub(r'\W+', '', x))
new_restaurant['Style'] = new_restaurant['Style'].map(lambda x: re.sub(r'\W+', '', x))
new_restaurant_db_array = convert_frame_to_array(new_restaurant)
def create_hash_column_test(new_restaurant_in):
    power_value = 256
    mod_value = 283  #100003
    name_hash = []
    address_hash = []
    phone_hash = []
    style_hash = []
    data_hash = []

    for row in new_restaurant_in:
        total_sum_column = 0
        count = 1
        for column in row:
            value_length = len(column)
            hash_value = 0
            column_hash = 0
            column = column.upper()
            for char in range(value_length):
                #print(column[char])
                hash_value = (hash_value * power_value + ord(column[char])) % mod_value
                column_hash = column_hash+hash_value
                total_sum_column += hash_value

            if count == 1:
                #print(column_hash)
                name_hash.append(column_hash)
                #print(name_hash)
            elif count == 2:
                address_hash.append(column_hash)
            elif count == 3:
                phone_hash.append(column_hash)
            elif count == 4:
                style_hash.append(column_hash)
            count+=1
        data_hash.append(total_sum_column)
    new_restaurant['NameHash'] = name_hash
    new_restaurant['AddressHash'] = address_hash
    new_restaurant['PhoneHash'] = phone_hash
    new_restaurant['StyleHash'] = style_hash
    new_restaurant['DataHash'] = data_hash
    return new_restaurant

create_hash_column_test(new_restaurant_db_array)

new_restaurant_hash = new_restaurant[['NameHash','AddressHash','PhoneHash','StyleHash']].copy()
for columns in new_restaurant_hash.columns:
    new_restaurant_hash[columns] = (new_restaurant[columns]-restaurant_db_graph[columns].min())/(restaurant_db_graph[columns].max()-restaurant_db_graph[columns].min())
print(restaurant_db_graph_normalised)
print(new_restaurant_hash)

cluster_labels = KMeans(n_clusters=10).fit(restaurant_db_graph_normalised)
labels = cluster_labels.labels_
print(labels)
predict_label = cluster_labels.predict(new_restaurant_hash)
print(predict_label)


GMM = GaussianMixture(n_components=10,covariance_type='full',max_iter=100, n_init=1, init_params='kmeans').fit(restaurant_db_graph_normalised)#restaurant_db['DataHash'])
print('Converged', GMM.converged_)
means = GMM.means_
covariances = GMM.covariances_

Y = np.array([[0.669466], [0.528786], [0.408916], [0.164902]])
prediction_data = GMM.predict(restaurant_db_graph_normalised)
prediction = GMM.predict(new_restaurant_hash)
print(prediction_data)
print(prediction)

