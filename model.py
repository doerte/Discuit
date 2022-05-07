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
absolute_features = ["correct"]


def run_all(data, n_sets, absolute, categorical, continuous, i):
    sign = False

    #get data from file
    dat = prepare_data(data, absolute, continuous, categorical)

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
        run_all(data, n_sets, absolute, categorical, continuous, ind)
    
    #give up trying after 20 runs
    elif sign:
        print("Ran 20 different models, none achieves a significance value of p>.2 on all features.")
        print("sets: ", sets)
        print(stats)
    
    #report outcome of succesful run
    else:
        #create .csv files including set allocation
        #make list with setname per input item
        set_numbers = []
        for item in inputD.index:
            for j in range(len(sets)):
                if item in sets[j]:
                    set_numbers.append(j+1)
            
        #add new column
        inputD['set_number'] = set_numbers

        #output file
        inputD.to_csv("output.csv", index=False)

        print("sets: ", sets)
        print(stats)

        # save statistics to file if there was more than 1 set
        if no_sets > 1:
            f = open("statistics.txt", "w")
            iterations = i+1
            stat_string = ("Number of iterations: %s \n \nResults of Kruskall-Wallis Anovas for the following variables:\n"  % iterations)
        
            for test in stats:
                stat_string += ("'" + stats[stats.index(test)][0] + "' (X2(%s) = %s, p = %s)" %(no_sets-1, round(stats[stats.index(test)][1],3), round(stats[stats.index(test)][2],3)) + ";\n")

            f.write(stat_string)
            f.close()

def prepare_data(data, absolute, continuous, categorical):
    #remove first column (item name)
    data = data.drop(data.columns[[0]], axis=1)
    #data = data.drop(columns=absolute) #only nec if not grouped first
   

    data_transformed = []

    #split by "absolute" feature and remove absolute features from clustering
    if len(absolute) > 0:
        grouped = data.groupby(absolute)
       
        for name, group in grouped:
            # drop absolute columns from further analysis
            data_x = group.drop(columns=absolute)
            # dummy-code categorical

            # transform data for fair comp continuous/categorical
            mms = MinMaxScaler()
            data_x[continuous] = mms.fit_transform(data_x[continuous])

            data_transformed.append(data_x)
    
    else:
        mms = MinMaxScaler()
        data[continuous] = mms.fit_transform(data[continuous])
        data_transformed.append(data)

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

    final_clusters = []

    for cluster in clusters:
        cluster_new = []
        for item in cluster:
            cluster_new.append(transformed_data.iloc[item].name)
        final_clusters.append(cluster_new)

    return final_clusters


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
    #statistics are still carried out over whole set, not over subparts according to absolute criterion
    stats = []

    if len(absolute_features) > 0:
        print("stats need adjustment")

        #get options in absolute column
        subsets = set(data[absolute_features[0]].tolist())
        
        #do stats per subset #still to be done
        #for subset in subsets:
        
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
    


    else:
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

if no_sets > 1:
    run_all(inputD, no_sets, absolute_features, categorical_features, continuous_features, 0)
else:
    print("Please use more than 1 set for this tool to be meaningful!")
