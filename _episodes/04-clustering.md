---
title: "Clustering with Scikit Learn"
teaching: 0
exercises: 0
questions:
- "Key question (FIXME)"
objectives:
- "Identify clusters in data using k-means clustering."
- "See the limitations of k-means when clusters overlap."
- "Use spectral clustering to overcome the limitations of k-means."
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
- "Unsupervised learning algorithms don't need training"
- "As well as providing machine learning algorithms scikit learn also has functions to make example data"
---

# Clustering

Clustering is the grouping of data points which are similar to each other. It can be a powerful technique for identifying patterns in data.
Clustering analysis does not usually require any training and is known as an unsupervised learning technique. The lack of a need for training
means it can be applied quickly 

## K-means clustering

The K-means clustering algorithm is a simple clustering algorithm that tries to identify the centre of each cluster.
It does this by searching for a point which minimises the distance between the centre and all the points in the cluster. 
The algorithm needs to be told how many clusters to look for, but a common technique is to try different numbers of clusters and combine
it with other tests to decide on the best combination.

### K-means with Scikit Learn

To perform a k-means clustering with Scikit learn we first need to import the sklearn.cluster module.

~~~
import sklearn.cluster as skl_cluster
~~~
{: .python}


> ## Working in multiple dimensions
> Although this example shows two dimensions the kmeans algorithm can work in more than two, it just becomes very difficult to show this visually
> once we get beyond 3 dimensions. Its very common in machine learning to be working with multiple variables and so our classifiers are working in
> multidimensonal spaces. 
{: .callout}

### Limitations of K-Means

* Requires number of clusters to be decided in advance
* Requires linear cluster boundaries

Fast to compute



## Spectral Clustering

Spectral clustering is a technique that attempts to overcome the linear boundary problem of k-means clustering. 
It works by treating clustering as a graph partitioning problem, its looking for nodes in a graph with a small distance between them.

see http://www.cvl.isy.liu.se:82/education/graduate/spectral-clustering/SC_course_part1.pdf for more details about how spectral clustering works.

Spectral clustering uses something called a kernel trick to introduce additional dimensions to the data. 
A common example of this is trying to cluster two almost overlapping crescent moon shapes or one circle within another.
A K-means classifier will fail to do this and will end up effectively drawing a line which crosses the cresecents/circles. 
Spectral clustering will introduce an additional dimension that effectively moves one of the cresecents (or circles) away from the other in the
additional dimension.

This has the downside of being more computationally expensive than k-means clustering.

### Spectral Clustering with Scikit Learn




