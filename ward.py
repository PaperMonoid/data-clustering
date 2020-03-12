from dataset import x1, x2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
import time

"""
https://scikit-learn.org/stable/modules/clustering.html
https://scikit-learn.org/stable/auto_examples/cluster/plot_coin_ward_segmentation.html#sphx-glr-auto-examples-cluster-plot-coin-ward-segmentation-py
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
https://stackoverflow.com/questions/29127013/plot-dendrogram-using-sklearn-agglomerativeclustering
https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/plot_hierarchical_clustering_dendrogram.py
https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
"""

print("Compute structured hierarchical clustering...")
st = time.time()

X = np.stack([x1, x2], axis=1)
X = np.reshape(X, (-1, 2))

n_clusters = 3
ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
ward.fit(X)

label = ward.labels_

print("Elapsed time: ", time.time() - st)
print("Number of clusters: ", np.unique(label).size)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter(x1, x2, c=label)

ax.set_xlabel(r"$x1$", fontsize=15)
ax.set_ylabel(r"$x2$", fontsize=15)
ax.set_title("Ward clustering $K = 3$ Scatter plot")

ax.grid(True)

plt.show()


def plot_dendrogram(model, **kwargs):
    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0] + 2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(
        float
    )

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# plot_dendrogram(ward, labels=ward.labels_)

linked = linkage(X, "ward")
label_list = ward.labels_
dendrogram(
    linked,
    orientation="top",
    labels=label_list,
    distance_sort="descending",
    show_leaf_counts=True,
)

plt.title("Ward Clustering $K=3$ Dendrogram")

plt.show()
