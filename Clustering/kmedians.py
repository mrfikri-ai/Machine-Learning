import numpy as np
import matplotlib.pyplot as plt

def kmedians(X, k, max_iterations=100):
    # Initialize k centroids randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    # Initialize cluster labels as zeros
    labels = np.zeros(X.shape[0], dtype=np.int)
    for i in range(max_iterations):
        # Assign each data point to its closest centroid
        for j, x in enumerate(X):
            distances = np.linalg.norm(centroids - x, ord=1, axis=1)
            labels[j] = np.argmin(distances)
        # Update centroids as the median of the data points in each cluster
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centroids[c] = np.median(X[mask], axis=0)
        # Check for convergence
        if i > 0 and np.all(labels == prev_labels):
            break
        prev_labels = labels.copy()
    # Return the centroids and labels
    return centroids, labels

# Example usage
X = np.random.randn(100, 2)
k = 6
centroids, labels = kmedians(X, k=k)
for c in range(3):
    print(f"Centroid {c}: {centroids[c]}")
    print(f"Points in cluster {c}:")
    print(X[labels == c])


def plot_clusters(X, centroids, labels):
    k = centroids.shape[0]
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    for c in range(k):
        mask = labels == c
        plt.scatter(X[mask, 0], X[mask, 1], color=colors[c], label=f"Cluster {c}")
        plt.scatter(centroids[c, 0], centroids[c, 1], marker='x', color='k')
    plt.legend()
    plt.show()

# Example usage
# X = np.random.randn(100, 2)
# centroids, labels = kmedians(X, k=3)
plot_clusters(X, centroids, labels)
