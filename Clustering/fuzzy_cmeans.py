def fuzzy_cmeans(points, k, max_iteration = 100):
        from scipy.spatial.distance import cdist
        
        # TODO: set the parameter
        m = 3

        # Define the variable of membership
        U = np.random.rand(points.shape[0],k)
        U /= np.sum(U, axis=1)[:,np.newaxis]

        # Declare the k centroids randomly
        def calculate_centroid (points, k, U, m):
            centroids = np.zeros((k, points.shape[1]))
            for i in range (k):
                centroids[i,:] = np.sum((U[:,i] ** m)[:,np.newaxis] * points, axis = 0) / np.sum(U[:,i] ** m)
            return centroids

        # Calculate new membership
        def calculate_membership (points, centroids, k , m):
            U_new = np.zeros((points.shape[0], k))
            for i in range (k):
                U_new[:,i] = np.linalg.norm(points - centroids[i,:], axis=1)

            U_new = 1 / (U_new ** (2/(m-1)) * np.sum((1/U_new) ** (2/(m-1)) , axis = 1 )[:, np.newaxis] )
            return U_new

        # Make the cluster has a better resolution
        for iteration in range (max_iteration):
            centroids = calculate_centroid(points, k, U , m)
            U_new = calculate_membership(points, centroids, k , m) 
      
            if np.linalg.norm (U_new - U) <= 0.00001:
                break
            
            U = U_new

        labels = np.argmax(U_new, axis=1)
        dict_dividedPoints = { i: points[labels == i] for i in range(k) }

        return centroids, dict_dividedPoints
