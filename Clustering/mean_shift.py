from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data using make_blobs
np.random.seed(42)
X, _ = make_blobs(n_samples=200, centers=4, n_features=2, random_state=42)

# Create and fit the mean-shift model
ms = MeanShift()
ms.fit(X)

# Get the cluster labels and cluster centers
cluster_labels = ms.labels_
cluster_centers = ms.cluster_centers_

# Plot the data points with cluster labels and cluster centers
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='red')
plt.title('Mean-Shift Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
