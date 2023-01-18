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
# TODO: work with missing data
# TODO: maybe include more than 1 absolute variable?

from typing import List

import pandas as pd
import sys
from scipy.stats import kruskal, chi2_contingency
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
                    choices=['l', 'c', 'n', 'a', 'd'],
                    help='provide the data type for each column (l(abel)/c(ategorical)'
                         '/n(umerical)/a(bsolute)/d(isregard).'
                         'The number of labels needs to match the number of columns'
                         ' in your input file. If this is not the case you can provide '
                         'them later on and your input will be ignored.'
                         '"Label" and "absolute" can only be specified once.',
                    default=None)
parser.add_argument('--runs', type=int,
                         help='indicate how many different output options you want to generate',
                    default=1)
args = parser.parse_args()
no_sets = int(sys.argv[2])

# read file and check if it's suitable
# noinspection PyBroadException
try:
    inputD = pd.read_csv(sys.argv[1])
    fileName = pathlib.Path(sys.argv[1]).with_suffix('').name
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
# number of runs provided as an argument. If nothing is provided it's 1. Also needs to come from GUI!
iterations = args.runs


# Check all the columns and ask about status. Label and absolute can only be chosen once.
if args.columns == None or len(args.columns) != len(inputD.columns):
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
        print(
            "More than one 'absolute' variable was specified. Please use -h to get help in providing suitable arguments")
        sys.exit(1)  # abort


def prepare_data(data, continuous, categorical, label, disregard):
    # remove label column & disregarded columns
    if len(label) != 0:
        data = data.drop([label[0]], axis=1)
    if len(disregard) != 0:
        data = data.drop(disregard, axis=1)
    # transform continuous data
    if len(continuous) != 0:
        # TODO replace md with average
        mms = MinMaxScaler()
        data[continuous] = mms.fit_transform(data[continuous])
    # make sure categorical data uses numbers (for silhouette score)
    if len(categorical) != 0:
        for feat in categorical:
            if data[feat].dtype != "float64" and data[feat].dtype != "int64":
                # find unique values
                values = data[feat].unique()
                i = 0
                # replace values
                for value in values:
                    data[feat].replace(value, i, inplace=True)
                    i=i+1
        # TODO handle missing data
    return data


def clustering(transformed_data, categorical_features, continuous_features):
    # determine max number of clusters...
    max_clus = int(len(transformed_data) * .5)
    if max_clus > 10:
        max_clus = 10
    cl_range = range(2, max_clus)  # changed to max 10 clusters to keep speed, check which max is appropriate
    # kmodes prototype for mixed numerical and categorical data
    largest_sil = (0, -1)

    # this needs to be adjusted depending on input
    categorical_features_idx = [transformed_data.columns.get_loc(col) for col in categorical_features]
    mark_array = transformed_data.values

    # choose algorithm depending on input
    if (len(categorical_features) != 0) and (len(continuous_features) != 0):
        for k in cl_range:
            kproto = KPrototypes(n_clusters=k, max_iter=20)
            kproto.fit_predict(mark_array, categorical=categorical_features_idx)
            sil = metrics.silhouette_score(transformed_data, kproto.labels_, sample_size=1000)

            if sil > largest_sil[1]:
                largest_sil = (k, sil)

        kproto_final = KPrototypes(n_clusters=largest_sil[0], max_iter=20)

        pred_cluster = kproto_final.fit_predict(mark_array, categorical=categorical_features_idx)

    elif (len(categorical_features) != 0) and (len(continuous_features) == 0):
        for k in cl_range:
            kmode = KModes(n_clusters=k, init="random", n_init=5)
            kmode.fit_predict(transformed_data)
            sil = metrics.silhouette_score(transformed_data, kmode.labels_, sample_size=1000)
            if sil > largest_sil[1]:
                largest_sil = (k, sil)
        kmode_final = KModes(n_clusters=largest_sil[0], init="random", n_init=5)
        pred_cluster = kmode_final.fit_predict(transformed_data)
    else:
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


def divide_in_sets(clusters, output_sets):
    # divide clusters evenly amongst desired sets
    for cluster in clusters:
        for item in cluster:
            output_sets[output_sets.index(min(output_sets, key=len))].append(item)


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


def kwtest(label, features, sets, data):
    stats = []
    df = len(sets) - 1
    for feat in features:
        kw_input = []
        for s_set in sets:
            list = data.loc[data.set_number == s_set, feat].tolist()
            kw_input.append(list)
        stat, p = kruskal(*kw_input)
        stats.append([label, "Kruskal-Wallis test", feat, stat, df, p])
    return stats


def chi(label, features, data):
    stats = []
    for feat in features:
        data_crosstab = pd.crosstab(data[feat],
                                    data['set_number'])
        stat, p, dof, expected = chi2_contingency(data_crosstab, correction=True)
        stats.append([label, "Chi2-Test", feat, stat, dof, p])
    return stats


def statistics(data):
    stats_out = []
    subsets = data[absolute_features[0]].unique()
    sets = data.set_number.unique()

    for subset in subsets:
        stats_frame = data.loc[data[absolute_features[0]] == subset]
        stats_out.append(kwtest(subset, continuous_features, sets, stats_frame))
        stats_out.append(chi(subset, categorical_features, stats_frame))

    # overall stats
    stats_out.append(kwtest("overall", continuous_features, sets, data))
    stats_out.append(chi("overall", categorical_features, data))
    return stats_out


def write_out(stats, i, significant, it_num):
    # output file
    outFileName = fileName + "_out" + str(it_num) + ".csv"
    inputD.to_csv(outFileName, index=False)
    # save statistics to file if there was more than 1 set
    if no_sets > 1:
        statFileName = fileName + "_stats" + str(it_num) + ".txt"
        f = open(statFileName, "w")
        iterations = i + 1
        stat_string = (
                "Number of iterations: %s \n \n"
                "Results for the following tests:\n" % iterations)

        if significant:
            stat_string += ("\nIn 20 iterations no split could be found that results in p>.2 for all variables.\n\n")

        for testgroup in stats:
            for test in testgroup:
                stat_string += ("Absolute variable instance '%s': " % (
                stats[stats.index(testgroup)][testgroup.index(test)][0])
                                + stats[stats.index(testgroup)][testgroup.index(test)][1] + ' for '
                                + stats[stats.index(testgroup)][testgroup.index(test)][2]
                                + ": X2(%s) = %s, p = %s" % (stats[stats.index(testgroup)][testgroup.index(test)][4],
                                                             round(stats[stats.index(testgroup)][testgroup.index(test)][
                                                                       3], 3),
                                                             round(stats[stats.index(testgroup)][testgroup.index(test)][
                                                                       5], 3)) + ";\n")


        if len(categorical_features) > 0:
            stat_string += ("\nCross-tables for the distribution of categorical features:\n\n")
            for feat in categorical_features:
                data_crosstab = pd.crosstab(inputD[feat],
                                            inputD['set_number'], margins=True)
                stat_string += (data_crosstab.to_string() + "\n\n")

        if len(absolute_features) > 0:
            stat_string += ("\nCross-table for the distribution of the absolute feature:\n\n")
            data_crosstab = pd.crosstab(inputD[absolute_features[0]],
                                        inputD['set_number'], margins=True)
            stat_string += (data_crosstab.to_string() + "\n\n")

        if len(continuous_features) > 0:
            stat_string += ("\nAverage values per set:\n\n")
            for feat in continuous_features:
                for set in range(1,no_sets+1):
                    mean = inputD.loc[inputD['set_number']== set , feat].mean()
                    stat_string += (feat + " in set " + str(set) +": " + str(mean) + "\n")

        f.write(stat_string)
        f.close()



def run_all(i, it_num):
    output_sets = []
    for single_set in range(0, no_sets):
        output_sets.append([])

    if no_sets > 1:
        # prepare data
        dat = prepare_data(inputD, continuous_features, categorical_features, label, disregard)

        # split by "absolute" feature and remove absolute features from clustering
        if len(absolute_features) == 1:
            datasets = split(absolute_features[0], dat)
        else:
            datasets = [dat]
    else:
        print("Please use more than 1 set for this tool to be meaningful!")
        sys.exit(1)  # abort

    # for each part of the absolute splitting make sets
    for data in datasets:
        # form clusters
        clusters = clustering(data, categorical_features, continuous_features)

        # divide in sets
        divide_in_sets(clusters, output_sets)

    set_numbers = []
    for item in inputD.index:
        for j in range(len(output_sets)):
            if item in output_sets[j]:
                set_numbers.append(j + 1)

    # add new column
    inputD['set_number'] = set_numbers

    # do statistics
    stats = statistics(inputD)

    # This checks for looping but is inside the loop
    all_ns = True

    for var_type in stats:
        for var in var_type:
            if var[5] < 0.2:
                all_ns = False

    # write to files
    if all_ns:
        write_out(stats, i, False, it_num)
    elif i < 19:
        i = i + 1
        run_all(i, it_num)
    else:
        print("failed")
        write_out(stats, i, True, it_num)


### actually run the program ###

for it_num in range(iterations):
    # progress bar
    perc = 20//iterations
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('=' * it_num * perc, it_num / iterations * 100))
    sys.stdout.flush()

    # initiate loop-tracking
    i = 0
    # start first loop
    run_all(i, it_num)

# final progress bar
sys.stdout.write('\r')
sys.stdout.write("[%-20s] %d%%" % ('=' * 20, 100))
sys.stdout.flush()

# initiate loop-tracking
#i = 0
# start first loop
#run_all(i)
