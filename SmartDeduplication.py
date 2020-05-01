import pandas as pd
import re
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from collections import Counter, defaultdict
from sklearn.decomposition import PCA

desired_width=320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns',10)

restaurant_db_cols = ['Name', 'Address', 'Phone', 'Style']
restaurant_db = pd.read_csv(r'CompleteRestaurantList.csv', sep=',', names=restaurant_db_cols, encoding="latin-1")
pd.set_option("display.max_rows", None, "display.max_columns", None)


restaurant_db['Name'] = restaurant_db['Name'].map(lambda x: re.sub(r'\W+', '', x))
restaurant_db['Address'] = restaurant_db['Address'].map(lambda x: re.sub(r'\W+', '', x))
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
    mod_value = 104729
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


def find_cluster_indices(cluster_no, cluster_labels):
    return np.where(cluster_labels == cluster_no)[0]


def find_duplicate_percentage(restaurant_db_array_cluster_wise):
    duplicate_phone = restaurant_db_array_cluster_wise[restaurant_db_array_cluster_wise.duplicated(['PhoneHash'],keep=False)]
    duplicate_phone = duplicate_phone.sort_values(by=['PhoneHash'])
    if duplicate_phone.empty:
        return 0,None
    else:

        duplicate_address = duplicate_phone[duplicate_phone.duplicated(['AddressHash'],keep = False)]
        duplicate_address = duplicate_address.sort_values(by=['AddressHash'])
        #print(len(restaurant_db_cols))
        if duplicate_address.empty:
            number_of_matching_columns = 1
            return number_of_matching_columns,duplicate_phone
            #print("Total percentage of duplicates in this cluster is: ", (len(duplicate_phone)/len(restaurant_db_array_cluster_wise))*100)
        else:
            duplicate_name = duplicate_address[duplicate_address.duplicated(['NameHash'],keep= False)]
            duplicate_name = duplicate_name.sort_values(by=['NameHash'])
            if duplicate_name.empty:
                number_of_matching_columns = 2
                remaining_matched_data = duplicate_phone[duplicate_phone['PhoneHash'].isin(duplicate_address)]
                print(remaining_matched_data)
                return number_of_matching_columns, duplicate_address
                #print("Total percentage of duplicates in this cluster is: ", (len(duplicate_address)/len(restaurant_db_array_cluster_wise))*100)
            else:
                duplicate_style = duplicate_name[duplicate_name.duplicated(subset=['PhoneHash','StyleHash'],keep= False)]
                if duplicate_style.empty:
                    number_of_matching_columns = 3
                    return number_of_matching_columns, duplicate_name
                    #print("Total percentage of duplicates in this cluster is: ", (len(duplicate_name))/len(restaurant_db_array_cluster_wise)  *100)
                else:
                    number_of_matching_columns = 4
                    return number_of_matching_columns, duplicate_style
                    #print("Total percentage of duplicates in this cluster is: ", ( len(duplicate_style)/len(restaurant_db_array_cluster_wise))*100)
"""
        print(duplicate_phone)
        test = restaurant_db_array_cluster_wise['PhoneHash'].isin(duplicate_phone['PhoneHash'])
        duplicate_address = restaurant_db_array_cluster_wise[test]
        print(duplicate_address)

        for row in duplicatePhone['PhoneHash']:
            dupe_rows = duplicatePhone.loc(row)
            print(dupe_rows)

        start_index = duplicatePhone.index[0]
        for index,row in duplicatePhone.iterrows():
            print(index)

            duplicate_index = binarySearch(duplicatePhone['PhoneHash'].drop([start_index,index]),row['PhoneHash'])
            print(duplicate_index)

def binarySearch(duplicate_frame,value):
    duplicate_list = duplicate_frame.values.tolist()
    duplicate_length = len(duplicate_list) - 1
    start = 0
    end = duplicate_length
    while start <= end:
        mid = (start+end)//2
        if duplicate_list[mid] == value:
            return mid
        if value>duplicate_list[mid]:
            start = mid+1
        else:
            end = mid-1
    if start>end:
        return None
"""
# Driver code to call the necessary functions


restaurant_db_array = convert_frame_to_array(restaurant_db)
create_hash_column(restaurant_db_array)
restaurant_db_array = convert_frame_to_array(restaurant_db)
restaurant_db_graph = restaurant_db[['NameHash','AddressHash', 'PhoneHash']].copy() #'NameHash', 'AddressHash', 'PhoneHash', 'StyleHash','DataHash'
restaurant_db_graph_normalised = restaurant_db_graph.copy()
restaurant_db_graph_normalised = normalise_data(restaurant_db_graph, restaurant_db_graph_normalised)
#print(restaurant_db_graph_normalised)
#find_number_of_cluster()

"""
pca = PCA()
restaurant_db_graph_normalised = pca.fit_transform(restaurant_db_graph_normalised)
explained_variance = pca.explained_variance_ratio_
print("explained_variance")
print(explained_variance)
"""

#Running the Testing file.

Name = 'Arts Deli'#input("Enter the Name of the restaurant: ")
Address = '12224 Ventura Blvd. Studio City'#input("Enter the address : ")
Phone = '404-875-0276'#input("Enter the phone number: ")
Style = 'Delis'#input("Enter the style: ")
new_restaurant = pd.DataFrame({"Name":[Name], "Address": [Address], "Phone":[Phone],"Style":[Style]})
new_restaurant['Name'] = new_restaurant['Name'].map(lambda x: re.sub(r'\W+', '', x))
new_restaurant['Address'] = new_restaurant['Address'].map(lambda x: re.sub(r'\W+', '', x))
new_restaurant['Phone'] = new_restaurant['Phone'].map(lambda x: re.sub(r'\W+', '', x))
new_restaurant['Style'] = new_restaurant['Style'].map(lambda x: re.sub(r'\W+', '', x))
new_restaurant_db_array = convert_frame_to_array(new_restaurant)
def create_hash_column_test(new_restaurant_in):
    power_value = 256
    mod_value = 283
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

new_restaurant_hash = new_restaurant[['NameHash','AddressHash','PhoneHash']].copy() #,'PhoneHash','StyleHash''NameHash','AddressHash','PhoneHash','StyleHash','DataHash'
for columns in new_restaurant_hash.columns:
    new_restaurant_hash[columns] = (new_restaurant[columns]-restaurant_db_graph[columns].min())/(restaurant_db_graph[columns].max()-restaurant_db_graph[columns].min())
print(restaurant_db_graph_normalised)
print(new_restaurant_hash)
def plot_data():
    fig, ax = plt.subplots()
    ax.scatter(restaurant_db['PhoneHash'], restaurant_db['DataHash'])
    ax.set_title('Dataset')
    ax.set_xlabel('PhoneHash')
    ax.set_ylabel('DataHash')
    plt.show()

    fig,ax1 = plt.subplots()
    ax1.hist( restaurant_db['DataHash'])
    ax1.set_title('Dataset')
    ax1.set_xlabel('DataHash')
    ax1.set_ylabel('Frequency')
    plt.show()


cluster_labels = KMeans(n_clusters=5).fit(restaurant_db_graph_normalised)
labels = cluster_labels.labels_
print(labels)
predict_label = cluster_labels.predict(new_restaurant_hash)#.drop(columns = ['Name', 'Address', 'Phone' , 'Style', 'DataHash']))
print(predict_label)
array_frame_cluster = {}
for cluster in range(cluster_labels.n_clusters):
    array_frame_cluster[cluster] = pd.DataFrame(restaurant_db_array[(find_cluster_indices(cluster,labels))],columns = ['Name', 'Address', 'Phone' , 'Style','NameHash','AddressHash','PhoneHash','StyleHash','DataHash'])
for key in array_frame_cluster:
    matching_columns,duplicate_df_obj = find_duplicate_percentage(array_frame_cluster[key])
    if matching_columns ==0:
        print("In Cluster: ",key,"No Matches Found")
    elif matching_columns == 1:
        print("In Cluster: ",key," Total % of data matched with other records is: ", round((1/len(restaurant_db_cols))*100),"and % of record that are matched over all is",round((len(duplicate_df_obj)/len(restaurant_db))*100))
        print("The matched records are:\n",duplicate_df_obj.drop(columns = ['NameHash','AddressHash','PhoneHash','StyleHash','DataHash']))
    elif matching_columns == 2:
        print("In Cluster: ",key,"Total % of data matched with other records is: ", round((2/len(restaurant_db_cols))*100),"and % of record that are matched over all is",round((len(duplicate_df_obj)/len(restaurant_db))*100))
        print("The matched records are:\n",duplicate_df_obj.drop(columns = ['NameHash','AddressHash','PhoneHash','StyleHash','DataHash']))
    elif matching_columns == 3:
        print("In Cluster: ",key,"Total % of data matched with other records is: ", round((3/len(restaurant_db_cols))*100),"and % of record that are matched over all is",round((len(duplicate_df_obj)/len(restaurant_db))*100))
        print("The matched records are:\n",duplicate_df_obj.drop(columns = ['NameHash','AddressHash','PhoneHash','StyleHash','DataHash']))
    else:
        print("In Cluster: ",key,"Total % of data matched with other records is: ", round((4 / len(restaurant_db_cols)) * 100),
              "and % of record that are matched over all is", round((len(duplicate_df_obj) / len(restaurant_db))* 100) )
        print("The matched records are\n:", duplicate_df_obj.drop(columns = ['NameHash','AddressHash','PhoneHash','StyleHash','DataHash']))

GMM = GaussianMixture(n_components=5).fit(restaurant_db_graph_normalised) #.drop(columns = 'DataHash'))#restaurant_db['DataHash'])max_iter=100, n_init=1, init_params='kmeans'
print('Converged', GMM.converged_)
means = GMM.means_
covariances = GMM.covariances_
#print(restaurant_db_graph)
prediction_data = GMM.predict(restaurant_db_graph_normalised) #.drop(columns = 'DataHash'))
prediction = GMM.predict(new_restaurant_hash) #.drop(columns = ['Name', 'Address', 'Phone' , 'Style', 'DataHash']))
print(prediction_data)
print(prediction)
