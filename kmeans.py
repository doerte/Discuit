from typing import List, Any

import pandas as pd
from scipy.stats import kruskal
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# to Do: ask which column should be evenly split (correctness?) Do that first
# to Do: ask to mark which columns are categorical

# must come from input
no_sets = 4
inputD = pd.read_csv("input.csv")
categorical_features = ["correct"]
continuous_features = ["freq", "aoa", "image"]
ignore_features = ["correct"]


def run_all(data, n_sets, ignore, categorical, continuous, i):
    sign = False
    dat = prepare_data(data, ignore, continuous, categorical)
    clusters = []
    for df in dat:
        clusters_df = clustering(df)
        clusters.append(clusters_df)

    sets = divide_in_sets(clusters, n_sets)
    stats = statistics(data, sets, continuous)
    for feat in stats:
        if feat[2] < 0.2:
            sign = True
            print(feat, " is too close to significance, run:", i)

    if sign and i < 20:
        ind = i + 1
        run_all(data, n_sets, ignore, categorical, continuous, ind)
    elif sign:
        print("Ran 20 different models, none achieves a significance value of p>.2 on all features.")
        print("sets: ", sets)
        print(stats)
    else:
        print("sets: ", sets)
        print(stats)


def prepare_data(data, ignore, continuous, categorical):
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
