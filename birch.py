from dataset import x1, x2
import numpy as np
from sklearn.cluster import Birch
import time

"""
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html
https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html#module-scipy.cluster.hierarchy
https://towardsdatascience.com/machine-learning-algorithms-part-12-hierarchical-agglomerative-clustering-example-in-python-1e18e0075019
https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
"""

print("Compute birch clustering...")
st = time.time()

X = np.stack([x1, x2], axis=1)
X = np.reshape(X, (-1, 2))

n_clusters = 3
birch = Birch(n_clusters=n_clusters, threshold=0.01, branching_factor=10)
birch.fit(X)

# label = birch.labels_
label = birch.predict(X)

print("Elapsed time: ", time.time() - st)
print("Number of clusters: ", np.unique(label).size)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter(x1, x2, c=label)

ax.set_xlabel(r"$x1$", fontsize=15)
ax.set_ylabel(r"$x2$", fontsize=15)
ax.set_title("Birch clustering $K = 3$ Scatter plot")

ax.grid(True)

plt.show()
