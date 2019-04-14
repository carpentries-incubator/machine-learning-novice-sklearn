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
data, cluster_id = skl_smpl.make_blobs(n_samples=8000, cluster_std=3,
                                       centers=4, random_state=1)

for cluster_count in range(2,11):
    model = skl_cluster.SpectralClustering(n_clusters=cluster_count,
                                       affinity='nearest_neighbors',
                                       assign_labels='kmeans')

    labels = model.fit_predict(data)
    
    plt.scatter(data[:, 0], data[:, 1], s=15, linewidth=0, c=labels)
    
    plt.title(str(cluster_count)+" Clusters")
    plt.show()

end_time = time.time()
print("Elapsed time = ", end_time-start_time, "seconds")