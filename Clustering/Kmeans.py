import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
    def fit(self, X):
        # Randomly initialize cluster centroids
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        # Iteratively update the centroids
        for i in range(self.max_iter):
            # Assign each data point to its nearest centroid
            clusters = [[] for _ in range(self.n_clusters)]
            for x in X:
                distances = np.linalg.norm(x - self.centroids, axis=1)
                cluster_idx = np.argmin(distances)
                clusters[cluster_idx].append(x)
            
            # Update the centroids to the mean of the data points in each cluster
            for j in range(self.n_clusters):
                if clusters[j]:
                    self.centroids[j] = np.mean(clusters[j], axis=0)
    
    def predict(self, X):
        # Predict the cluster labels for the data points
        distances = np.linalg.norm(X - self.centroids[:, np.newaxis], axis=2)
        labels = np.argmin(distances, axis=0)
        return labels

# Generating random data points
X = np.random.rand(100, 2)

# Initializing KMeans object
kmeans = KMeans(n_clusters=3)

# Fitting the KMeans model to the data
kmeans.fit(X)

# Getting the cluster labels
labels = kmeans.predict(X)

# Getting the cluster centroids
centroids = kmeans.centroids

# Printing the cluster labels and centroids
print("Cluster labels: ", labels)
print("Cluster centroids: ", centroids)
