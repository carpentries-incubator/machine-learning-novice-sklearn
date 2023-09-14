---
title: "Unsupervised methods - Dimensionality reduction"
teaching: 10
exercises: 10
questions:
- How do we apply machine learning techniques to data with higher dimensions?
objectives:
- "Recall that most data is inherently multidimensional."
- "Understand that reducing the number of dimensions can simplify modelling and allow classifications to be performed."
- "Apply Principle Component Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE) to reduce the dimensions of data."
- "Evaluate the relative peformance of PCA and t-SNE in reducing data dimensionality."
keypoints:
- "PCA is a linear dimensionality reduction technique for tabular data"
- "t-SNE is another dimensionality reduction technique for tabular data that is more general than PCA"
---

# Dimensionality reduction

As seen in the last episode, general clustering algorithms work well with low-dimensional data. In this episode we will work with higher-dimension data such as images of handwritten text or numbers. The dataset we will be using is the Scikit-Learn subset of the Modified National Institute of Standards and Technology (MNIST) dataset. The MNIST dataset contains 70,000 images of handwritten numbers from 0-9, labelled with the number they contain, whereas our Scikit-Learn example is a smaller sample of 1797 of these images. An illustration of MNIST dataset is presented below as an example. 

![MNIST example illustrating all the classes in the dataset](../fig/MnistExamples.png)

In addition to taking a subset of the MNIST images, the Scikit-Learn images are compressed down to 8x8 pixels in size, rather than the MNIST 32x32, resulting in 64 pixel values per image. Each pixel of the Scikit-Learn images takes a value between 0-16. The dataset takes the form of a 1797x64 sized array. Each row corresponds to an image for which we have a label of 0-9, and each column corresponds to a flattened array of the 64 pixels, each of which can be considered a feature (or "dimension") of our data. 

The code to retrieve the entire dataset with Scikit-Learn is given below:

~~~
from sklearn import datasets

digits = datasets.load_digits()

# Examine the dataset
print(digits.data)
print(digits.target)

x = digits.data
y = digits.target

print(x.shape, y.shape)
y = digits.target
~~~
{: .language-python}


Linear clustering approaches such as k-means would require all the images to be binned into a pre-determined number of clusters, which might not adequately capture the variability in the images. 

Non-linear clustering, such as spectral clustering, might fare better, but requires the images to be projected into a higher dimension space. Separating the existing complex data in higher-order spaces would require complex non-linear separators to divide this data and really increase the computational cost of doing this.

We can help reduce the computational cost of clustering by transforming our higher-dimension input dataset into lower-order projections. Conceptually this is done by determining which combination of variables explain the most variance in the data, and then working with those variables. 


## Dimensionality reduction with Scikit-Learn
We will look at two commonly used techniques for dimensionality reduction: Principal Component Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE). Both of these techniques are supported by Scikit-Learn.

### Principal Component Analysis (PCA)

PCA uses feature extraction, i.e., it combines our input variables in a specific way so we can drop the least important or least significant variables while still retaining the fundamental attributes of our old variables. 

PCA takes the variance(or spread) of the data into account to reduce the dimensions.  Dimensions or variables having high variance have high information. Therefore, variables having very low variance can be removed or skipped. 

When the variance of both the dimensions is comparable, i.e., there is no significant difference in the variance of the dimensions we are trying to compare, PCA projects the data from the two dimensions into a single vector which is the direction of the maximum variance. Let's understand this with an example.

More complicated explanation: PCA is an important technique for visualizing high-dimensional data. PCA is a deterministic linear technique which projects n-dimensional data into k-dimensions, where `n < k`, while preserving the global structure of the input data. Dimension reduction in PCA involves transforming highly-correlated data by rotating the original set of vectors into their principal components. The variance between the transformation can be inferred by computing their eigen values.

The process of reducing dimensionality in PCA is as follows,
1. calculate the mean of each column
2. center the value in each column by subtracting the mean column value
3. calculate covariance matrix of centered matrix
4. calculate eigendecomposition of the covariance (eigenvectors represent the magnitude of directions or components of the subspace reduction)

Minimizing the eigen values closer to zero implies that the dataset has been successfully decomposed into it's respective principal components. 

Scikit-Learn lets us apply PCA in a relatively simple way. Lets code and apply PCA to the MNIST dataset. 

~~~
from sklearn import decomposition

# PCA
pca = decomposition.PCA(n_components=2)
pca.fit(x)
x_pca = pca.transform(x)

print(x_pca.shape)
~~~
{: .language-python}

This returns us an array of 1797x2 where the 2 remaining columns(our new "features" or "dimensions") contain vector representations of the First principle components (column 0) and Second Principle components (column 1) for each of the images. We can plot these two new features against each other:

~~~
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(1, figsize=(4, 4))

tx = x_pca[:, 0]
ty = x_pca[:, 1]

plt.scatter(tx, ty, c=y, cmap=plt.cm.nipy_spectral, 
        edgecolor='k',label=y)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.show()
~~~
{: .language-python}

![Reduction using PCA](../fig/pca.svg)

As illustrated in the figure above, PCA does not handle outlier data well, primarily due to global preservation of structural information. Pre-determining the principal components also has some of the same drawbacks as k-means clustering approaches. 

### t-distributed Stochastic Neighbor Embedding (t-SNE)
t-SNE is a non-deterministic non-linear technique which involves several optional hyper-parameters such as perplexity, learning rate, and number of steps. While the t-SNE algorithm is complex to explain, it works on the principle of preserving local similarities by minimizing the pairwise gaussian distance between two or more points in high-dimensional space. The versatility of the algorithm in transforming the underlying structural information into lower-order projections makes t-SNE applicable to a wide range of research domains.

Scikit-Learn allows us to apply t-SNE in a relatively simple way. Lets code and apply t-SNE to the MNIST dataset.

~~~
from sklearn import manifold

# t-SNE embedding
tsne = manifold.TSNE(n_components=2, init='pca', random_state = 0)
x_tsne = tsne.fit_transform(x)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, cmap=plt.cm.nipy_spectral,
        edgecolor='k',label=y)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.show()
~~~
{: .language-python}

![Reduction using t-SNE](../fig/tsne.svg)

The major drawback of applying t-SNE to datasets is the large computational requirement. Furthermore, hyper-parameter tuning of t-SNE usually requires some trial and error to perfect. In the above figure, the algorithm still has trouble separating all the classes perfectly. To account for even higher-order input data, neural networks were developed to more accurately extract feature information.


> ## Exercise: Working in three dimensions
> The above example has considered only two dimensions since humans
> can visualize two dimensions very well. However, there can be cases
> where a dataset requires more than two dimensions to be appropriately
> decomposed. Modify the above programs to use three dimensions and 
> create appropriate plots.
> Do three dimensions allow one to better distinguish between the digits?
>
> > ## Solution
> > ~~~
> > from mpl_toolkits.mplot3d import Axes3D
> > # PCA
> > pca = decomposition.PCA(n_components=3)
> > pca.fit(x)
> > x_pca = pca.transform(x)
> > fig = plt.figure(1, figsize=(4, 4))
> > ax = fig.add_subplot(projection='3d')
> > ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=y,
> >           cmap=plt.cm.nipy_spectral, s=9, lw=0)
> > plt.show()
> > ~~~
> > {: .language-python}
> >
> > ![Reduction to 3 components using pca](../fig/pca_3d.svg)
> >
> > ~~~
> > # t-SNE embedding
> > tsne = manifold.TSNE(n_components=3, init='pca',
> >         random_state = 0)
> > x_tsne = tsne.fit_transform(x)
> > fig = plt.figure(1, figsize=(4, 4))
> > ax = fig.add_subplot(projection='3d')
> > ax.scatter(x_tsne[:, 0], x_tsne[:, 1], x_tsne[:, 2], c=y,
> >           cmap=plt.cm.nipy_spectral, s=9, lw=0)
> > plt.show()
> > ~~~
> > {: .language-python}
> >
> > ![Reduction to 3 components using tsne](../fig/tsne_3d.svg)
> >
> >
> {: .solution}
{: .challenge}

> ## Exercise: Parameters
>
> Look up parameters that can be changed in PCA and t-SNE,
> and experiment with these. How do they change your resulting
> plots?  Might the choice of parameters lead you to make different
> conclusions about your data?
{: .challenge}

> ## Exercise: Other algorithms
>
> There are other algorithms that can be used for doing dimensionality
> reduction (for example the Higher Order Singular Value Decomposition (HOSVD)).
> Do an internet search for some of these and
> examine the example data that they are used on. Are there cases where they do 
> poorly? What level of care might you need to use before applying such methods
> for automation in critical scenarios?  What about for interactive data 
> exploration?
{: .challenge}

{% include links.md %}

