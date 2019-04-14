#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib
import mpl_toolkits.mplot3d as plt3d
import sklearn.cluster as skl_cluster
import sklearn.datasets as skl_data
import numpy as np


def plot_3d_separation(circles,labels):

    #create two empty arrays for the points inside a list
    circles2 = [np.zeros([200,3]),np.zeros([200,3])]

    circle_indicies = [0,0]
    for i in range(0,400):
        x = circles[i][0]
        y = circles[i][1]
        label = labels[i]
        j = circle_indicies[label]
        circles2[label][j][0] = x
        circles2[label][j][1] = y
        circles2[label][j][2] = label
        circle_indicies[label] = circle_indicies[label] + 1

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(circles2[0][:, 0],circles2[0][:, 1],circles2[0][:, 2],c=matplotlib.cm.flag(0))
    ax.scatter(circles2[1][:, 0],circles2[1][:, 1],circles2[1][:, 2],c=matplotlib.cm.flag(255))

    plt.show()


def cluster_circles():

    circles, circles_clusters = skl_data.make_circles(n_samples=400, noise=.01, random_state=0)

  #  plt.scatter(circles[:, 0], circles[:, 1], s=5, c=circles_clusters)

    Kmean = skl_cluster.KMeans(n_clusters=2)

    # run the kmeans algorithm on the data
    Kmean.fit(circles)

    # use the kmeans predictor to work out which class each item is in
    clusters = Kmean.predict(circles)

    # plot the data, colouring it by cluster
    plt.scatter(circles[:, 0], circles[:, 1], s=15, linewidth=0.1, c=clusters,cmap='flag')
    plt.show()

    model = skl_cluster.SpectralClustering(n_clusters=2,
                                           affinity='nearest_neighbors',
                                           assign_labels='kmeans')

    labels = model.fit_predict(circles)
    print(labels)
    plt.scatter(circles[:, 0], circles[:, 1], s=15, linewidth=0, c=labels, cmap='flag')
    plt.show()
    # uncomment to show separation in 3D
    # plot_3d_separation(circles,labels)

cluster_circles()