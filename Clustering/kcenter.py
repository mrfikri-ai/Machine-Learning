import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

def k_center_clustering(X, k):
    n_samples, n_features = X.shape

    # Initialize cluster centers
    centers = [X[0]]  # Start with the first data point as the initial center

    while len(centers) < k:
        # Compute distances between data points and cluster centers
        distances = cdist(X, centers)
        min_distances = np.min(distances, axis=1)

        # Select the data point with the maximum minimum distance as the new center
        new_center_index = np.argmax(min_distances)
        new_center = X[new_center_index]
        centers.append(new_center)

    # Assign each data point to the nearest cluster center
    labels = np.argmin(cdist(X, centers), axis=1)

    return centers, labels

# Generate sample data using make_blobs
n_samples = 200
centers = 4
X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=42)

# Perform k-center clustering
k = 6
centers, labels = k_center_clustering(X, k)

# Plot the data points and cluster centers
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(np.array(centers)[:, 0], np.array(centers)[:, 1], c='white', marker='x', label='Cluster Centers')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Center Clustering')
plt.show()
