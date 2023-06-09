def kmedians(points, k, max_iterations=1000):
        from scipy.spatial.distance import cdist

        # Initialize k centroids randomly
        centroids = points[np.random.choice(range(len(points)), k, replace=False)]
        # Initialize cluster labels as zeros
        
        for _ in range(max_iterations):
            # Assign each data point to its closest centroid
            distances = cdist(points, centroids, metric='cityblock')
            labels = np.argmin(distances, axis=1)

            # Update the cluster centers using medians
            new_centers = np.array([np.median(points[labels == i], axis=0) for i in range(k)])

            # Check convergence
            if np.array_equal(centroids, new_centers):
                break

            centroids = new_centers

            # prev_labels = labels.copy()
            dict_dividedPoints = { i: points[labels == i] for i in range(k) }
        # Return the centroids and labels
        return centroids, dict_dividedPoints
