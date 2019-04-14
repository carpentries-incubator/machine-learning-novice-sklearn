#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 02:08:24 2019

@author: colin
"""

import matplotlib.pyplot as plt
import sklearn.cluster as skl_cluster
import sklearn.datasets.samples_generator as skl_smpl
import time


start_time = time.time()
data, cluster_id = skl_smpl.make_blobs(n_samples=800000, cluster_std=3,
                                       centers=4, random_state=1)

for cluster_count in range(2,11):
    Kmean = skl_cluster.KMeans(n_clusters=cluster_count)
    
    #run the kmeans algorithm on the data
    Kmean.fit(data)
    
    # use the kmeans predictor to work out which class each item is in
    clusters = Kmean.predict(data)
    
    plt.scatter(data[:, 0], data[:, 1], s=15, linewidth=0, c=clusters)
    
    plt.title(str(cluster_count)+" Clusters")
    plt.show()

end_time = time.time()
print("Elapsed time = ", end_time-start_time, "seconds")