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
from scipy.stats import kruskal
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
import argparse
import pathlib

# check whether path and number of sets arguments were provided
parser = argparse.ArgumentParser()
parser.add_argument('datapath', type=pathlib.Path, help='path to input data file (csv)')
parser.add_argument('sets', type=int, help='provide number of desired sets')
parser.add_argument('--columns', nargs='*',
                    choices=['l', 'c', 'n', 'a', 'd'], help='list column types',
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


def run_all(data, n_sets, absolute, categorical, continuous, label, disregard, i):
    sign = False

    # get data from file
    dat = prepare_data(data, absolute, continuous, label, disregard)

    # form clusters
    clusters = []

    for df in dat:
        clusters_df = clustering(df, categorical, continuous)
        clusters.append(clusters_df)

    # divide in sets
    sets = divide_in_sets(clusters, n_sets)

    # compare sets to each other statistically
    stats = statistics(data, sets, continuous, absolute)

    for feat in stats:
        if feat[2] < 0.2:
            sign = True
            print(feat, " is too close to significance, run:", i)

    # run again if sets are not different enough (max. 20 times)
    if sign and i < 20:
        ind = i + 1
        run_all(data, n_sets, absolute, categorical, continuous, label, disregard, ind)

    # give up trying after 20 runs
    elif sign:
        print("Ran 20 different models, none achieves a significance value of p>.2 on all features.")
        print("sets: ", sets)
        print(stats)

    # report outcome of successful run
    else:
        # create .csv files including set allocation
        # make list with set name per input item
        set_numbers = []
        for item in inputD.index:
            for j in range(len(sets)):
                if item in sets[j]:
                    set_numbers.append(j + 1)

        # add new column
        inputD['set_number'] = set_numbers

        # output file
        inputD.to_csv("output.csv", index=False)

        print("sets: ", sets)
        print(stats)

        # save statistics to file if there was more than 1 set
        if no_sets > 1:
            f = open("statistics.txt", "w")
            iterations = i + 1
            stat_string = (
                    "Number of iterations: %s \n \n"
                    "Results of Kruskal-Wallis Anovas for the following variables:\n" % iterations)
            # add reporting of numbers per category per set

            for test in stats:
                stat_string += ("'" + stats[stats.index(test)][0] + "' (X2(%s) = %s, p = %s)" % (
                    no_sets - 1, round(stats[stats.index(test)][1], 3), round(stats[stats.index(test)][2], 3)) + ";\n")

            f.write(stat_string)
            f.close()


def prepare_data(data, absolute, continuous, label, disregard):
    # remove label column & disregarded columns
    if len(label) != 0:
        data = data.drop([label[0]], axis=1)
    if len(disregard) != 0:
        data = data.drop(disregard, axis=1)
    data_transformed = []

    # split by "absolute" feature and remove absolute features from clustering
    if len(absolute) > 0:
        try:
            grouped = data.groupby(absolute)
        except KeyError:
            print('You listed an absolute variable that cannot be found in the input file')
            sys.exit(1)  # abort

        for name, group in grouped:
            # drop absolute columns from further analysis
            data_x = group.drop(columns=absolute)
            # dummy-code categorical

            # transform data for fair comp continuous/categorical
            mms = MinMaxScaler()

            if len(continuous) != 0:
                try:
                    data_x[continuous] = mms.fit_transform(data_x[continuous])
                except KeyError:
                    print("You listed (a) numerical variable/s that cannot be found in the input file")
                    sys.exit(1)  # abort

            data_transformed.append(data_x)

    else:
        if len(continuous) != 0:
            mms = MinMaxScaler()
            data[continuous] = mms.fit_transform(data[continuous])
        data_transformed.append(data)

    return data_transformed


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

    for cl in clusters:
        for cluster in cl:
            for item in cluster:
                sets[sets.index(min(sets, key=len))].append(item)
    return sets


def statistics(data, sets, features, absolute_features):
    # statistics are still carried out over whole set, not over sub-parts according to absolute criterion
    # maybe do statistics for categorical variables as well?

    if len(absolute_features) > 0:
        print("stats need adjustment")

        # get options in absolute column
        subsets = set(data[absolute_features[0]].tolist())
        print(subsets)
        # todo: do stats per subset
        # for subset in subsets:
        stats = kwtest(features, sets, data)

    else:
        stats = kwtest(features, sets, data)
    return stats


def kwtest(features, sets, data):
    stats = []
    for feat in features:
        kw_input = []
        for s_set in sets:
            sub_set = []
            for item in s_set:
                sub_set.append(data.loc[item, feat])
            kw_input.append(sub_set)
        stat, p = kruskal(*kw_input)
        stats.append([feat, stat, p])
    return stats


if no_sets > 1:
    run_all(inputD, no_sets, absolute_features, categorical_features, continuous_features, label, disregard, 0)
else:
    print("Please use more than 1 set for this tool to be meaningful!")
