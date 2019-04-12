#!/usr/bin/env python3
import matplotlib.pyplot as plt
import sklearn.cluster as skl_cluster
import sklearn.datasets as skl_data


def cluster_circles():

    circles, circles_clusters = skl_data.make_circles(n_samples=400, noise=.01, random_state=0)

  #  plt.scatter(circles[:, 0], circles[:, 1], s=5, c=circles_clusters)

    Kmean = skl_cluster.KMeans(n_clusters=2)

    # run the kmeans algorithm on the data
    Kmean.fit(circles)

    # use the kmeans predictor to work out which class each item is in
    clusters = Kmean.predict(circles)

    # plot the data, colouring it by cluster
   # plt.scatter(circles[:, 0], circles[:, 1], s=5, c=clusters)

    model = skl_cluster.SpectralClustering(n_clusters=2,
                                           affinity='nearest_neighbors',
                                           assign_labels='kmeans')

    labels = model.fit_predict(circles)
#    plt.scatter(circles[:, 0], circles[:, 1], s=5, c=labels)

    import numpy as np


    #create two empty arrays for the points inside a list
    circles2 = [np.zeros([200,2]),np.zeros([200,2])]

    circle_indicies = [0,0]
    for i in range(0,400):
        x = circles[i][0]
        y = circles[i][1]
        label = labels[i]
        j = circle_indicies[label]
        circles2[label][j][0] = x
        circles2[label][j][1] = y
        circle_indicies[label] = circle_indicies[label] + 1


    print(circles2[0])

    plt.scatter(circles2[1][:, 0], circles2[1][:, 1], s=5)

cluster_circles()