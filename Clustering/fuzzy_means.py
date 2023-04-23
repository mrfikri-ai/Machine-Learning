# TODO: Still under investigation

import numpy as np

class FuzzyKMeans:
    def __init__(self, n_clusters=2, m=2, max_iter=100, tol=0.0001):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        centers = self._initialize_centers(X)
        iteration = 0
        while iteration < self.max_iter:
            # Calculate the membership matrix
            membership_mat = self._calculate_membership(X, centers)
            # Update the cluster centers
            new_centers = self._calculate_centers(X, membership_mat)
            # Check for convergence
            if np.linalg.norm(new_centers - centers) < self.tol:
                break
            centers = new_centers
            iteration += 1
        self.labels_ = np.argmax(membership_mat, axis=1)
        self.cluster_centers_ = centers

    def _initialize_centers(self, X):
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _calculate_membership(self, X, centers):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:,i] = np.linalg.norm(X - centers[i], axis=1)
        membership_mat = np.zeros((X.shape[0], self.n_clusters))
        for i in range(X.shape[0]):
            for j in range(self.n_clusters):
                membership_mat[i,j] = np.sum([np.power((distances[i,j] / distances[i,k]), 2/(self.m-1)) for k in range(self.n_clusters)])
            membership_mat[i,:] = 1 / membership_mat[i,:]
        return membership_mat

    def _calculate_centers(self, X, membership_mat):
        membership_mat = np.power(membership_mat, self.m)
        centers = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            centers[i,:] = np.sum(X * membership_mat[:,i].reshape(-1,1), axis=0) / np.sum(membership_mat[:,i])
        return centers

# # Call the class above
# from sklearn.datasets import make_blobs

# # Generate sample data
# X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# # Fit the fuzzy K-means model
# fkm = FuzzyKMeans(n_clusters=3)
# fkm.fit(X)

# # Get the cluster labels and centers
# labels = fkm.labels_
# centers = fkm.cluster_centers_

# print("Cluster labels:")
# print(labels)

# print("Cluster centers:")
# print(centers)
