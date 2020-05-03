import pandas as pd
import re
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

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


def create_hash_column(restaurant_db_array_in,df_name):
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
    add_hash_to_df(name_hash,address_hash,phone_hash,style_hash,data_hash,df_name)


def add_hash_to_df(name_hash,address_hash,phone_hash,style_hash,data_hash,df_name):
    if df_name == 0:
        restaurant_db['NameHash'] = name_hash
        restaurant_db['AddressHash'] = address_hash
        restaurant_db['PhoneHash'] = phone_hash
        restaurant_db['StyleHash'] = style_hash
        restaurant_db['DataHash'] = data_hash
    else:
        new_restaurant['NameHash'] = name_hash
        new_restaurant['AddressHash'] = address_hash
        new_restaurant['PhoneHash'] = phone_hash
        new_restaurant['StyleHash'] = style_hash
        new_restaurant['DataHash'] = data_hash


def normalise_data(restaurant_db_norm, restaurant_db_normalised_df):
    for feature in restaurant_db_norm.columns:
        max_value = restaurant_db_graph[feature].max()
        min_value = restaurant_db_graph[feature].min()
        restaurant_db_normalised_df[feature] = (restaurant_db_graph[feature]-min_value)/(max_value-min_value)
    return restaurant_db_normalised_df


def find_number_of_cluster():
    distorsions = []
    for k in range(2, 10):
        kmeans_cluster = KMeans(n_clusters=k)
        kmeans_cluster.fit(restaurant_db_graph_normalised)
        distorsions.append(kmeans_cluster.inertia_)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, 10), distorsions)
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
                #print(remaining_matched_data)
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





# Driver code to call the necessary functions
#choosing your approach for EM clustering or KMeans clustering
#print("Select the algorithm for clustering.\nChoose \n1. KMeans \n2.EMClustering")
#choice = int(input(""))

restaurant_db_array = convert_frame_to_array(restaurant_db)
create_hash_column(restaurant_db_array,0)
restaurant_db_array = convert_frame_to_array(restaurant_db)

restaurant_db_graph = restaurant_db[['NameHash','AddressHash', 'PhoneHash']].copy() #'NameHash', 'AddressHash', 'PhoneHash', 'StyleHash','DataHash'
restaurant_db_graph_normalised = restaurant_db_graph.copy()
restaurant_db_graph_normalised = normalise_data(restaurant_db_graph, restaurant_db_graph_normalised)
#print(restaurant_db_graph_normalised)
#find_number_of_cluster()


"""
plot_data()

pca = PCA()
restaurant_db_graph_normalised = pca.fit_transform(restaurant_db_graph_normalised)
explained_variance = pca.explained_variance_ratio_
print("explained_variance")
print(explained_variance)
"""

#Running the Testing file.


#def duplicate_detector_for_cross_validation(array_frame_cluster):
def cross_validation_dataset(restaurant_db_array):
    kf = KFold(n_splits=5)
    kf.get_n_splits(restaurant_db_array)
    #print(kf)
    count = 1
    for train_index, test_index in kf.split(restaurant_db_array):
#Uncomment this section:
        #print("\n Cross validation number: ",count)
        count = count+1
        #print("Train:",train_index,"Test",test_index)
        restaurant_db_array_train, restaurant_db_array_test = restaurant_db_array[train_index], restaurant_db_array[test_index]
        for i in range(2):

            if i ==0:

                #Uncomment this section print("\n Kmeans deduplicated data")
                total_duplicate_df_obj = []
                predict_label = kmeans_clustering(train_index, test_index)
            else:
                #Uncomment this section print("\n EM deduplicated data")
                total_duplicate_df_obj = []
                predict_label = em_clustering(train_index, test_index)
            array_frame_cluster = {}
            for cluster in np.unique(predict_label):
                #array_frame_cluster[cluster] = pd.DataFrame(restaurant_db_array[(find_cluster_indices(cluster,labels))],columns = ['Name', 'Address', 'Phone' , 'Style','NameHash','AddressHash','PhoneHash','StyleHash','DataHash'])
                array_frame_cluster[cluster] = pd.DataFrame(restaurant_db_array_test[(find_cluster_indices(cluster, predict_label))],columns=['Name', 'Address', 'Phone', 'Style', 'NameHash','AddressHash', 'PhoneHash', 'StyleHash', 'DataHash'])

            for key in array_frame_cluster:
                matching_columns, duplicate_df_obj = find_duplicate_percentage(array_frame_cluster[key])
                if duplicate_df_obj is None:
                    continue
                else:
                    total_duplicate_df_obj.append(duplicate_df_obj)
            # Uncomment this section if len(total_duplicate_df_obj) != 0:

                #Uncomment this section print("\n Concatenated duplicates list of Cross Validation: \n", pd.concat(total_duplicate_df_obj))


#def find_record_matching():


"""
            total_duplicate_df_obj = []
            duplicate_detector_for_cross_validation(array_frame_cluster)
            for key in array_frame_cluster:
                matching_columns,duplicate_df_obj = find_duplicate_percentage(array_frame_cluster[key])
                if duplicate_df_obj is None:
                    continue
                else:
                    total_duplicate_df_obj.append(duplicate_df_obj)

                if matching_columns ==0:
                    continue
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
            if not total_duplicate_df_obj:
                continue
            else:
                print("Concatenated list",pd.concat(total_duplicate_df_obj))
"""
def single_duplicate_checker(predict_label,labels):
    labels = np.hstack((labels, np.atleast_1d(predict_label).T))
    #print(labels)
    #labels = np.column_stack((labels,predict_label))
    array_frame_cluster = pd.DataFrame(restaurant_db_array[(find_cluster_indices(predict_label, labels))],
                                       columns=['Name', 'Address', 'Phone', 'Style', 'NameHash', 'AddressHash',
                                                'PhoneHash', 'StyleHash', 'DataHash'])
    duplicate_phone = array_frame_cluster[array_frame_cluster.duplicated(['PhoneHash'], keep=False)]

    check_phone = new_restaurant['PhoneHash']
    df = duplicate_phone.where(duplicate_phone.values == new_restaurant.values)

    df = df.dropna(thresh=2)
    df = df.drop(columns=['NameHash', 'AddressHash',
                                  'PhoneHash', 'StyleHash', 'DataHash'])

    output = duplicate_phone.loc[df.index]
    if len(output)>0:
        print("The records which have at least 50% of match are: \n",output.drop(columns=['NameHash', 'AddressHash',
                                  'PhoneHash', 'StyleHash', 'DataHash']))
    else:
        print("No Match found!!!")


def kmeans_clustering(train_index,test_index):
    kmeans_algo = KMeans(n_clusters=5).fit(restaurant_db_graph_normalised.iloc[train_index])
    labels = kmeans_algo.labels_
    # print(labels)
    #predict_df = restaurant_db_graph_normalised.iloc[test_index].copy()
    if len(restaurant_db_graph_normalised.iloc[test_index]) == 3:
        predict_array = (restaurant_db_graph_normalised.iloc[test_index]).to_numpy()
        predict_array = predict_array.reshape(1,-1)
        predict_label = kmeans_algo.predict(predict_array)
        single_duplicate_checker(predict_label,labels)
    else:
        predict_label = kmeans_algo.predict(restaurant_db_graph_normalised.iloc[test_index])
        return predict_label


def em_clustering(train_index,test_index):
    GMM = GaussianMixture(n_components=5).fit(restaurant_db_graph_normalised.iloc[train_index]) #.drop(columns = 'DataHash'))#restaurant_db['DataHash'])max_iter=100, n_init=1, init_params='kmeans'
    predict_gmm =  GMM.predict(restaurant_db_graph_normalised.iloc[train_index])
    if len(restaurant_db_graph_normalised.iloc[test_index]) == 3:
        predict_array = (restaurant_db_graph_normalised.iloc[test_index]).to_numpy()
        predict_array = predict_array.reshape(1,-1)
        predict_label = GMM.predict(predict_array)
        single_duplicate_checker(predict_label,predict_gmm)
        #print(GMM.means_)
        #print(GMM.covariances_)
    else:
        prediction = GMM.predict(restaurant_db_graph_normalised.iloc[test_index]) #.drop(columns = ['Name', 'Address', 'Phone' , 'Style', 'DataHash']))
        return prediction

"""
cluster_labels = KMeans(n_clusters=5).fit(restaurant_db_graph_normalised)
labels = cluster_labels.labels_

print("\nKMEANS RESULTS:")
#predict_label = cluster_labels.predict(new_restaurant_hash)#.drop(columns = ['Name', 'Address', 'Phone' , 'Style', 'DataHash']))
#print(predict_label)
array_frame_cluster = {}
for cluster in range(cluster_labels.n_clusters):
    array_frame_cluster[cluster] = pd.DataFrame(restaurant_db_array[(find_cluster_indices(cluster,labels))],columns = ['Name', 'Address', 'Phone' , 'Style','NameHash','AddressHash','PhoneHash','StyleHash','DataHash'])
for key in array_frame_cluster:
    matching_columns,duplicate_df_obj = find_duplicate_percentage(array_frame_cluster[key])
    if matching_columns ==0:

        print("\nIn Cluster: ",key,"No Matches Found")
    elif matching_columns == 1:
        print("\nIn Cluster: ",key," Total % of data matched with other records is: ", round((1/len(restaurant_db_cols))*100),"and % of record that are matched over all is",round((len(duplicate_df_obj)/len(restaurant_db))*100))
        print("\nThe matched records are:\n",duplicate_df_obj.drop(columns = ['NameHash','AddressHash','PhoneHash','StyleHash','DataHash']))
    elif matching_columns == 2:
        print("\nIn Cluster: ",key,"Total % of data matched with other records is: ", round((2/len(restaurant_db_cols))*100),"and % of record that are matched over all is",round((len(duplicate_df_obj)/len(restaurant_db))*100))
        print("\nThe matched records are:\n",duplicate_df_obj.drop(columns = ['NameHash','AddressHash','PhoneHash','StyleHash','DataHash']))
    elif matching_columns == 3:
        print("\nIn Cluster: ",key,"Total % of data matched with other records is: ", round((3/len(restaurant_db_cols))*100),"and % of record that are matched over all is",round((len(duplicate_df_obj)/len(restaurant_db))*100))
        print("\nThe matched records are:\n",duplicate_df_obj.drop(columns = ['NameHash','AddressHash','PhoneHash','StyleHash','DataHash']))
    else:
        print("\nIn Cluster: ",key,"Total % of data matched with other records is: ", round((4 / len(restaurant_db_cols)) * 100),
              "and % of record that are matched over all is", round((len(duplicate_df_obj) / len(restaurant_db))* 100) )
        print("\nThe matched records are\n:", duplicate_df_obj.drop(columns = ['NameHash','AddressHash','PhoneHash','StyleHash','DataHash']))
GMM = GaussianMixture(n_components=5).fit(restaurant_db_graph_normalised)  # .drop(columns = 'DataHash'))#restaurant_db['DataHash'])max_iter=100, n_init=1, init_params='kmeans'
predict_gmm = GMM.predict(restaurant_db_graph_normalised)
print("\nEM ALgorithm  RESULTS:")
array_frame_cluster = {}
for cluster in range(cluster_labels.n_clusters):
    array_frame_cluster[cluster] = pd.DataFrame(restaurant_db_array[(find_cluster_indices(cluster,predict_gmm))],columns = ['Name', 'Address', 'Phone' , 'Style','NameHash','AddressHash','PhoneHash','StyleHash','DataHash'])
for key in array_frame_cluster:
    matching_columns,duplicate_df_obj = find_duplicate_percentage(array_frame_cluster[key])
    if matching_columns ==0:

        print("\nIn Cluster: ",key,"No Matches Found")
    elif matching_columns == 1:
        print("\nIn Cluster: ",key," Total % of data matched with other records is: ", round((1/len(restaurant_db_cols))*100),"and % of record that are matched over all is",round((len(duplicate_df_obj)/len(restaurant_db))*100))
        print("\nThe matched records are:\n",duplicate_df_obj.drop(columns = ['NameHash','AddressHash','PhoneHash','StyleHash','DataHash']))
    elif matching_columns == 2:
        print("\nIn Cluster: ",key,"Total % of data matched with other records is: ", round((2/len(restaurant_db_cols))*100),"and % of record that are matched over all is",round((len(duplicate_df_obj)/len(restaurant_db))*100))
        print("\nThe matched records are:\n",duplicate_df_obj.drop(columns = ['NameHash','AddressHash','PhoneHash','StyleHash','DataHash']))
    elif matching_columns == 3:
        print("\nIn Cluster: ",key,"Total % of data matched with other records is: ", round((3/len(restaurant_db_cols))*100),"and % of record that are matched over all is",round((len(duplicate_df_obj)/len(restaurant_db))*100))
        print("\nThe matched records are:\n",duplicate_df_obj.drop(columns = ['NameHash','AddressHash','PhoneHash','StyleHash','DataHash']))
    else:
        print("\nIn Cluster: ",key,"Total % of data matched with other records is: ", round((4 / len(restaurant_db_cols)) * 100),
              "and % of record that are matched over all is", round((len(duplicate_df_obj) / len(restaurant_db))* 100) )
        print("\nThe matched records are\n:", duplicate_df_obj.drop(columns = ['NameHash','AddressHash','PhoneHash','StyleHash','DataHash']))
"""


#cross_validation_dataset(restaurant_db_array)

"""   
cluster_labels = KMeans(n_clusters=5).fit(restaurant_db_graph_normalised.iloc[train_index])
labels = cluster_labels.labels_
print(labels)
predict_label = cluster_labels.predict(new_restaurant_hash)#.drop(columns = ['Name', 'Address', 'Phone' , 'Style', 'DataHash']))
print(predict_label)
"""

train_index = np.arange(0,(len(restaurant_db_graph_normalised.index)))

Name = input("Enter the Name of the restaurant: ") #'Arts Deli'#
Address = input("Enter the address : ") #'12224 Ventura Blvd. Studio City'#
Phone = input("Enter the phone number: ") #'404-875-0276'#
Style = input("Enter the style: ") #'Delis'#
new_restaurant = pd.DataFrame({"Name":[Name], "Address": [Address], "Phone":[Phone],"Style":[Style]})
new_restaurant['Name'] = new_restaurant['Name'].map(lambda x: re.sub(r'\W+', '', x))
new_restaurant['Address'] = new_restaurant['Address'].map(lambda x: re.sub(r'\W+', '', x))
new_restaurant['Phone'] = new_restaurant['Phone'].map(lambda x: re.sub(r'\W+', '', x))
new_restaurant['Style'] = new_restaurant['Style'].map(lambda x: re.sub(r'\W+', '', x))

new_restaurant_db_array = convert_frame_to_array(new_restaurant)
create_hash_column(new_restaurant_db_array,1)
new_restaurant_db_array = convert_frame_to_array(new_restaurant)
restaurant_db_array = np.vstack((restaurant_db_array,new_restaurant_db_array))
#print(new_restaurant)
new_restaurant_hash = new_restaurant[['NameHash','AddressHash','PhoneHash']].copy() #,'PhoneHash','StyleHash''NameHash','AddressHash','PhoneHash','StyleHash','DataHash'
for columns in new_restaurant_hash.columns:
    new_restaurant_hash[columns] = (new_restaurant[columns]-restaurant_db_graph[columns].min())/(restaurant_db_graph[columns].max()-restaurant_db_graph[columns].min())
restaurant_db_graph_normalised = restaurant_db_graph_normalised.append(new_restaurant_hash,ignore_index = True)
test_index = np.array(len(restaurant_db_graph_normalised.index)-1)
#if choice == 1:
print("The output predicted by KMeans Algorithm:")
kmeans_clustering(train_index,test_index)
#elif choice == 2:
print("The output predicted by EM clustering Algorithm:")
em_clustering(train_index,test_index)
#else:print("Enter a valid choice")

