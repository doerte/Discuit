## Copyright 2022 Dörte de Kok
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.


from typing import List, Any

import pandas as pd
import sys
from scipy.stats import kruskal
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# to Do: ask which column should be evenly split (correctness?) Do that first
# to Do: more than one categorical variable...

# must come from input. In GUI this should be selected in the GUI after opening a file!
try:
    no_sets = int(sys.argv[2])
except:
    print('You failed to provide the number of required sets'\
        'on the command line! Your input should look as follows: '\
        'model.py [path to csv file] [number of sets].')
    sys.exit(1) #abort
try:
    inputD = pd.read_csv(sys.argv[1])
except:
    print('You failed to provide your input file (.csv) '\
          'on the command line! Your input should look as follows: '\
        'model.py [path to csv file] [number of sets].')
    sys.exit(1)  # abort

categorical_features = ["wordclass"]
continuous_features = ["freq", "image"]
ignore_features = ["correct"]


def run_all(data, n_sets, ignore, categorical, continuous, i):
    sign = False

    #get data from file
    dat = prepare_data(data, ignore, continuous, categorical)

    #form clusters
    clusters = []
    for df in dat:
        clusters_df = clustering(df)
        clusters.append(clusters_df)

    #divide in sets
    sets = divide_in_sets(clusters, n_sets)
    
    #compare sets to each other statistically
    stats = statistics(data, sets, continuous)
    for feat in stats:
        if feat[2] < 0.2:
            sign = True
            print(feat, " is too close to significance, run:", i)

    #run again if sets are not different enough (max. 20 times)
    if sign and i < 20:
        ind = i + 1
        run_all(data, n_sets, ignore, categorical, continuous, ind)
    
    #give up trying after 20 runs
    elif sign:
        print("Ran 20 different models, none achieves a significance value of p>.2 on all features.")
        print("sets: ", sets)
        print(stats)
    
    #report outcome of succesful run
    else:
        #create .csv files for new sets
        counter = 0
        for item_set in sets:
            counter = counter + 1
            temp_set = inputD.iloc[item_set]
            file_name = "output_set_" + str(counter) + ".csv"
            temp_set.to_csv(file_name)


        print("sets: ", sets)
        print(stats)
        #add: save statistics to file

        #BUG: seems like items area in more than 1 set... first 15 double than 4 single, 20-35 missing


def prepare_data(data, ignore, continuous, categorical):
    #remove first column (item name)
    data = data.drop(data.columns[[0]], axis=1)
   
    # split by "ignore" feature and remove ignored features from clustering
    grouped = data.groupby('correct')
    data_transformed = []

    for name, group in grouped:
        # drop ignored columns from further analysis
        data_x = group.drop(columns=ignore)
        # dummy-code categorical

        # transform data for fair comp continuous/categorical
        mms = MinMaxScaler()
        data_x[continuous] = mms.fit_transform(data_x[continuous])

        data_transformed.append(data_x)
    return data_transformed

def clustering(transformed_data):
    cl_range = range(2, len(transformed_data))
    # k means determine k
    largest_sil = (0, 0)
    clusters = []

    for k in cl_range:
        km = KMeans(n_clusters=k, n_init=1, init='k-means++')
        km.fit_predict(transformed_data)
        sil = metrics.silhouette_score(transformed_data, km.labels_, sample_size=1000)
        if sil > largest_sil[1]:
            largest_sil = (k, sil)
    km_final = KMeans(n_clusters=largest_sil[0], init='k-means++', n_init=1)
    pred_cluster = km_final.fit_predict(transformed_data)

    for k in range(0, largest_sil[0]):
        clusters.append([])

    for item in range(0, len(pred_cluster)):
        clusters[pred_cluster[item]].append(item)

    return clusters


def divide_in_sets(clusters, n_sets):
    sets = []
    for single_set in range(0, n_sets):
        sets.append([])

    for cl in clusters:
        for cluster in cl:
            for item in cluster:
                sets[sets.index(min(sets, key=len))].append(item)
    return sets


def statistics(data, sets, features):
    stats = []

    for feat in features:
        kw_input = []
        for s_set in sets:
            sub_set = []
            for item in s_set:
                sub_set.append(data.loc[item, feat])
            kw_input.append(sub_set)

        args = kw_input
        stat, p = kruskal(*args)
        stats.append([feat, stat, p])
    return stats


run_all(inputD, no_sets, ignore_features, categorical_features, continuous_features, 0)
