# Copyright 2022 Dörte de Kok
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List

import pandas as pd
import sys
from scipy.stats import kruskal, fisher_exact
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
import argparse
import pathlib

# Maybe reconsider set-up for absolute feature. If absolute is present, run everything ("run all") twice instead of
# working around things later. Only thing to consider: how to merge datafiles in the end before making output-files?
# That part needs to happen in an outer procedure
#This seems to be working now.
#TODO: check whether absolute feature is binary! Or come up with something to handle non-binary
#TODO: write statistics to file
#TODO: make sure each set has same number of items (with second absolute thing start backwards?)

#check whether path and number of sets arguments were provided
parser = argparse.ArgumentParser()
parser.add_argument('datapath', type=pathlib.Path, help='path to input data file (csv)')
parser.add_argument('sets', type=int, help='provide number of desired sets')
parser.add_argument('--columns', nargs='*',
                    choices=['l', 'c', 'n', 'a', 'd'], help='provide the data type for each column (l(abel)/c(ategorical)'
                                                            '/n(umerical)/a(bsolute)/d(isregard).'
                                                            'The number of labels needs to match the number of columns'
                                                            ' in your input file. If this is not the case you can provide '
                                                            'them later on and your input will be ignored.'
                                                            '"Label" and "absolute" can only be specified once.',
                    default=None)
args = parser.parse_args()
no_sets = int(sys.argv[2])


# read file and check if it's suitable
# noinspection PyBroadException
try:
    inputD = pd.read_csv(sys.argv[1])
except FileNotFoundError:
    print("File not found.")
    sys.exit(1)  # abort
except pd.errors.EmptyDataError:
    print("No data")
    sys.exit(1)  # abort
except pd.errors.ParserError:
    print("Parse error")
    sys.exit(1)  # abort
except Exception:
    print("Something else went wrong. \n "
          "Make sure your input looks as follows: \n"
          "'model.py [path to csv file] [number of sets].'")
    sys.exit(1)  # abort

# The following info must come from user. In GUI this should be selected in the GUI after opening a file!
categorical_features = []
continuous_features = []
absolute_features = []
label = []
disregard = []

# Check all the columns and ask about status. Label and absolute can only be chosen once.
if len(args.columns) != len(inputD.columns):
    print("You didn't provide valid data type indications when running the program. Please specify them now")
    for column in inputD.columns:
        feature = None
        while feature is None:
            input_value = input("Is '" + column + "' the label (can only be assigned once), a categorical, "
                                                  "numerical or absolute (can be assigned once) variable "
                                                  "or should it be disregarded in splitting? l/c/n/a/d ")
            if input_value not in ('l', 'c', 'n', 'a', 'd'):
                print("Please choose either l, c, n, a or d ")
            else:
                feature = input_value
                if feature == "c":
                    categorical_features.append(column)
                elif feature == "n":
                    continuous_features.append(column)
                elif feature == "a":
                    if len(absolute_features) > 0:
                        print('You already have an absolute feature. Please choose something else.')
                        feature = None
                    else:
                        absolute_features.append(column)
                elif feature == "l":
                    if len(label) > 0:
                        print('You already have a label. Please choose something else.')
                        feature = None
                    else:
                        label.append(column)
                elif feature == "d":
                    disregard.append(column)
# if specified when running program, take them from there
else:
    for column in inputD.columns:
        feature = args.columns[inputD.columns.get_loc(column)]
        if feature == "c":
            categorical_features.append(column)
        elif feature == "n":
            continuous_features.append(column)
        elif feature == "a":
            absolute_features.append(column)
        elif feature == "l":
            label.append(column)
        elif feature == "d":
            disregard.append(column)
    if len(label) > 1:
        print("More than one 'label' was specified. Please use -h to get help in providing suitable arguments")
        sys.exit(1)  # abort
    if len(absolute_features) > 1:
        print("More than one 'absolute' variable was specified. Please use -h to get help in providing suitable arguments")
        sys.exit(1)  # abort



def run_all(data, n_sets, categorical, continuous):
    sign = False

    # form clusters
    clusters = clustering(data, categorical, continuous)

    # divide in sets
    sets = divide_in_sets(clusters, n_sets)
    return sets

def prepare_data(data, continuous, label, disregard):
    # remove label column & disregarded columns
    if len(label) != 0:
        data = data.drop([label[0]], axis=1)
    if len(disregard) != 0:
        data = data.drop(disregard, axis=1)
    # transform continuous data
    if len(continuous) != 0:
        mms = MinMaxScaler()
        data[continuous] = mms.fit_transform(data[continuous])
    return data


def clustering(transformed_data, categorical_features, continuous_features):
    cl_range = range(2, 10)  # changed to max 10 clusters to keep speed, check which max is appropriate

    # kmodes prototype for mixed numerical and categorical data
    largest_sil = (0, 0)

    # this needs to be adjusted depending on input
    categorical_features_idx = [transformed_data.columns.get_loc(col) for col in categorical_features]
    mark_array = transformed_data.values

    # choose algorithm depending on input
    if (len(categorical_features) != 0) and (len(continuous_features) != 0):
        print("both features")
        for k in cl_range:
            kproto = KPrototypes(n_clusters=k, max_iter=20)
            kproto.fit_predict(mark_array, categorical=categorical_features_idx)
            sil = metrics.silhouette_score(transformed_data, kproto.labels_, sample_size=1000)
            if sil > largest_sil[1]:
                largest_sil = (k, sil)
        kproto_final = KPrototypes(n_clusters=largest_sil[0], max_iter=20)
        pred_cluster = kproto_final.fit_predict(mark_array, categorical=categorical_features_idx)
    elif (len(categorical_features) != 0) and (len(continuous_features) == 0):
        print("only cat features")
        for k in cl_range:
            kmode = KModes(n_clusters=k, init="random", n_init=5)
            kmode.fit_predict(transformed_data)
            sil = metrics.silhouette_score(transformed_data, kmode.labels_, sample_size=1000)
            if sil > largest_sil[1]:
                largest_sil = (k, sil)
        kmode_final = KModes(n_clusters=largest_sil[0], init="random", n_init=5)
        pred_cluster = kmode_final.fit_predict(transformed_data)
    else:
        print("only num features")
        for k in cl_range:
            km = KMeans(n_clusters=k, n_init=1, init='k-means++')
            km.fit_predict(transformed_data)
            sil = metrics.silhouette_score(transformed_data, km.labels_, sample_size=1000)
            if sil > largest_sil[1]:
                largest_sil = (k, sil)
        km_final = KMeans(n_clusters=largest_sil[0], init='k-means++', n_init=1)
        pred_cluster = km_final.fit_predict(transformed_data)

    clusters: List[List[int]] = [[] for _ in range(largest_sil[0])]

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

    for cluster in clusters:
        for item in cluster:
            sets[sets.index(min(sets, key=len))].append(item)
    return sets

def split(absolute, data):
    try:
        grouped = data.groupby(absolute)
    except KeyError:
        print('You listed an absolute variable that cannot be found in the input file')
        sys.exit(1)  # abort

    data_splitted = []
    for name, group in grouped:
        # drop absolute columns from further analysis
        data_x = group.drop(columns=absolute)
        data_splitted.append(data_x)

    return data_splitted

### actually run the program ###
output_sets = []
stats = []
abs_labels = []

if no_sets > 1:
    # prepare data
    count = 0
    dat = prepare_data(inputD, continuous_features, label, disregard)

    # split by "absolute" feature and remove absolute features from clustering
    if len(absolute_features) == 1:
        abs_labels = list(set(dat[absolute_features[0]].tolist()))
        datasets = split(absolute_features[0],dat)
    else:
        datasets = [dat]
else:
    print("Please use more than 1 set for this tool to be meaningful!")
    sys.exit(1) # abort

#for each part of the absolute splitting run all including statistics
for data in datasets:
    #clustering + dividing in sets
    sets = run_all(data, no_sets, categorical_features, continuous_features)
    #add to list of sets for later reintegration to overall outcome sets
    output_sets.append(sets)

# merge outcome into one file and report
final_sets = []
for k in range(no_sets):
    joined_set = output_sets[0][k] + output_sets[1][k]
    final_sets.append(joined_set)

set_numbers = []
for item in inputD.index:
    for j in range(len(final_sets)):
        if item in final_sets[j]:
            set_numbers.append(j + 1)

# add new column
inputD['set_number'] = set_numbers

# output file
inputD.to_csv("output.csv", index=False)


# # do statistics for complete thing
# # do statistics continuous variables (are they different between sets for this part of absolute split)
# statsSet = kwtest(continuous_features, sets, data)
# # add info on which part of abs split this concerns
# k = 0
# statsSet.append(abs_labels[k])
# k = k + 1
#
# # do statistics categorical variables
# # count instances of each categFeat and compare between sets
# # Maybe move to actual stats method later on, input required
# # categorical_features, data, sets
#
# # for feat in categorical_features:
# #    instances = data[feat].tolist()
# #    uniq_instances = {*instances}
# #    for instance in uniq_instances:
# #        count = data.count(instance)
#
#
# fisher_out = fisher(categorical_features, sets, data)
#
# # append outcome cat variables to statsSet
#
# # note down stats outcomes for later report
# stats.append(statsSet)
#
# def statistics(data, sets, features, absolute_features):
#     # statistics are still carried out over whole set, not over sub-parts according to absolute criterion
#     # maybe do statistics for categorical variables as well?
#
#     if len(absolute_features) > 0:
#         print("stats need adjustment")
#         stats = kwtest(features, sets, data) #remove once other stats are inplace
#         ## implementation for absolute splitting below, but still raises problem in data output later on,
#         ## maybe rewrite (see top of file)
#         ## get options in absolute column
#         # stats = []
#         #subsets = set(data[absolute_features[0]].tolist())
#         ## TODO: do stats per subset
#         #for subset in subsets:
#         #    subset1 = data[data[absolute_features[0]] == subset]
#         #    working_set = [list(filter(lambda x: x in list(subset1.index.values), sublist)) for sublist in sets]
#         #    print(features,working_set, subset1)
#         #    stats1 = kwtest(features, working_set, subset1)
#         #    stats.append(stats1)
#         #print(stats)
#
#     else:
#         stats = kwtest(features, sets, data)
#     return stats
#
# def kwtest(features, sets, data):
#     stats = []
#     for feat in features:
#         kw_input = []
#         for s_set in sets:
#             sub_set = []
#             for item in s_set:
#                 sub_set.append(data.loc[item, feat])
#             kw_input.append(sub_set)
#         stat, p = kruskal(*kw_input)
#         stats.append([feat, stat, p])
#     return stats
#
# def fisher(features, sets, data):
#     stats = []
#
#     for feat in features:
#         for s_set in sets:
#             for item in s_set:
#                 print(feat)
#                 print(item, data.loc[item, feat])
#         #print(chi_input)
#
#
#
#     #data = [[10, 4],
#     #        [4, 9]]
    #print(fisher_exact(data))


# compare sets to each other statistically
# stats = statistics(data, sets, continuous, absolute)
#
# # this doesn't work yet for split statistics, as the output of stats is [[[0,1,2],[0,1,2]],[[0,1,2],[0,1,2]]]
# # instead of [[0,1,2],[0,1,2]]
# for feat in stats:
#     if feat[2] < 0.2:
#         sign = True
#         print(feat, " is too close to significance, run:", i)
#
# # run again if sets are not different enough (max. 20 times)
# if sign and i < 20:
#     ind = i + 1
#     run_all(data, n_sets, absolute, categorical, continuous, label, disregard, ind)
#
# # give up trying after 20 runs
# elif sign:
#     print("Ran 20 different models, none achieves a significance value of p>.2 on all features.")
#     print("sets: ", sets)
#     print(stats)
#
# # report outcome of successful run
# else:
#     # create .csv files including set allocation
#     # make list with set name per input item
#     set_numbers = []
#     for item in inputD.index:
#         for j in range(len(sets)):
#             if item in sets[j]:
#                 set_numbers.append(j + 1)
#
#     # add new column
#     inputD['set_number'] = set_numbers
#
#     # output file
#     inputD.to_csv("output.csv", index=False)
#
#     print("sets: ", sets)
# print(stats)
#
# # save statistics to file if there was more than 1 set
# if no_sets > 1:
#     f = open("statistics.txt", "w")
#     iterations = i + 1
#     stat_string = (
#             "Number of iterations: %s \n \n"
#             "Results of Kruskal-Wallis Anovas for the following variables:\n" % iterations)
#     # add reporting of numbers per category per set
#
#     for test in stats:
#         stat_string += ("'" + stats[stats.index(test)][0] + "' (X2(%s) = %s, p = %s)" % (
#             no_sets - 1, round(stats[stats.index(test)][1], 3), round(stats[stats.index(test)][2], 3)) + ";\n")
#
#     f.write(stat_string)
#     f.close()