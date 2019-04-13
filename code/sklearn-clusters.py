#!/usr/bin/env python3
import matplotlib.pyplot as plt
import sklearn.cluster as skl_cluster
import sklearn.datasets.samples_generator as skl_smpl


def cluster_blobs():
    data, cluster_id = skl_smpl.make_blobs(n_samples=400, cluster_std=0.75,
                                           centers=4, random_state=1)

    Kmean = skl_cluster.KMeans(n_clusters=4)

    # run the kmeans algorithm on the data
    Kmean.fit(data)

    # use the kmeans predictor to work out which class each item is in
    clusters = Kmean.predict(data)

    # plot the data, colouring it by cluster
    plt.scatter(data[:, 0], data[:, 1], s=5, linewidth=0,c=clusters)

    # plot the centres of each cluster as an X
    for cluster_x, cluster_y in Kmean.cluster_centers_:
        plt.scatter(cluster_x, cluster_y, s=100, c='r', marker='x')

    plt.show()

cluster_blobs()
