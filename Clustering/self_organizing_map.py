import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

def initialize_weights(input_dim, output_dim):
    """
    Initialize the weights for the SOM.

    Parameters:
    - input_dim: Number of input features.
    - output_dim: Shape of the SOM grid.

    Returns:
    - weights: Initial weight matrix of shape (output_dim[0], output_dim[1], input_dim).
    """
    weights = np.random.randn(output_dim[0], output_dim[1], input_dim)
    return weights

def find_best_matching_unit(sample, weights):
    """
    Find the best matching unit (BMU) for a given sample.

    Parameters:
    - sample: Input sample to be matched.
    - weights: SOM weight matrix.

    Returns:
    - bmu_index: Indices of the BMU in the weight matrix.
    """
    distances = cdist(sample[np.newaxis, :], weights.reshape(-1, weights.shape[2]))
    bmu_index = np.argmin(distances)
    return np.unravel_index(bmu_index, weights.shape[:2])

def update_weights(sample, weights, bmu_index, learning_rate, radius):
    """
    Update the weights of the SOM based on the best matching unit (BMU).

    Parameters:
    - sample: Input sample.
    - weights: SOM weight matrix.
    - bmu_index: Indices of the BMU in the weight matrix.
    - learning_rate: Learning rate for weight update.
    - radius: Neighborhood radius for weight update.
    """
    rows, cols = weights.shape[:2]
    bmu_row, bmu_col = bmu_index

    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - bmu_row) ** 2 + (j - bmu_col) ** 2)
            if dist <= radius:
                influence = np.exp(-dist / (2 * (radius ** 2)))
                weights[i, j] += learning_rate * influence * (sample - weights[i, j])

def self_organizing_maps(X, output_dim, num_epochs, learning_rate_initial, radius_initial):
    """
    Perform self-organizing maps (SOM) clustering.

    Parameters:
    - X: Input data array of shape (n_samples, n_features).
    - output_dim: Shape of the SOM grid.
    - num_epochs: Number of training epochs.
    - learning_rate_initial: Initial learning rate.
    - radius_initial: Initial neighborhood radius.

    Returns:
    - weights: Final SOM weight matrix.
    - cluster_labels: Cluster labels for each input sample.
    """
    n_samples, n_features = X.shape
    weights = initialize_weights(n_features, output_dim)
    learning_rate = learning_rate_initial
    radius = radius_initial

    for epoch in range(num_epochs):
        for sample in X:
            bmu_index = find_best_matching_unit(sample, weights)
            update_weights(sample, weights, bmu_index, learning_rate, radius)

        # Decay learning rate and radius
        learning_rate *= 0.9
        radius *= 0.9

    # Assign cluster labels based on the BMU for each sample
    cluster_labels = np.zeros(n_samples, dtype=int)
    for i, sample in enumerate(X):
        bmu_index = find_best_matching_unit(sample, weights)
        cluster_labels[i] = np.ravel_multi_index(bmu_index, output_dim)

    return weights, cluster_labels

# Generate sample data using make_blobs
np.random.seed(42)
X, _ = make_blobs(n_samples=200, centers=4, n_features=2, random_state=42)

# Set SOM parameters
output_dim = (10, 10)  # Shape of the SOM grid
num_epochs = 100  # Number of training epochs
learning_rate_initial = 0.5  # Initial learning rate
radius_initial = max(output_dim) / 2  # Initial neighborhood radius

# Perform self-organizing maps clustering
weights, cluster_labels = self_organizing_maps(X, output_dim, num_epochs, learning_rate_initial, radius_initial)

# Plot the clusters and SOM grid
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(weights[:, :, 0], weights[:, :, 1], marker='x', color='red')
plt.title('Self-Organizing Maps Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
