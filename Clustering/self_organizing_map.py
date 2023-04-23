# TODO: Fix the error in the self organizing maps

import numpy as np
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, input_dim, output_dim, lr=0.1, sigma=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.sigma = sigma if sigma is not None else np.sqrt(input_dim**2 + output_dim**2)/2
        
        self.weights = np.random.rand(output_dim[0], output_dim[1], input_dim[0], input_dim[1])
        
    def get_bmu(self, input_data):
        dist = np.sum((self.weights - input_data)**2, axis=(2,3))
        return np.unravel_index(np.argmin(dist), dist.shape)
    
    def update_weights(self, bmu, input_data, epoch):
        lr_decay = 1 - epoch / n_epochs
        sigma_decay = self.sigma * lr_decay
        lr_cur = self.lr * lr_decay
        for i in range(self.output_dim[0]):
            for j in range(self.output_dim[1]):
                dist = np.sqrt((i-bmu[0])**2 + (j-bmu[1])**2)
                if dist <= sigma_decay:
                    influence = np.exp(-dist**2 / (2*sigma_decay**2))
                    self.weights[i][j] += lr_cur * influence * (input_data - self.weights[i][j])
                    
    def train(self, data, n_epochs):
        for epoch in range(n_epochs):
            for d in data:
                bmu = self.get_bmu(d)
                self.update_weights(bmu, d, epoch)
                
    def predict(self, data):
        predictions = []
        for d in data:
            bmu = self.get_bmu(d)
            predictions.append(bmu)
        
        labels = np.zeros(len(data), dtype=np.int)
        centroids = np.zeros((self.output_dim[0]*self.output_dim[1], self.input_dim[0]*self.input_dim[1]))
        
        for i, p in enumerate(predictions):
            labels[i] = p[0] * self.output_dim[1] + p[1]
            centroids[labels[i]] += data[i]
        
        for i in range(len(centroids)):
            if np.sum(labels == i) > 0:
                centroids[i] /= np.sum(labels == i)
        
        return labels, centroids
        
    def plot_weights(self):
        plt.imshow(np.vstack([np.hstack(self.weights[i,j]) for i in range(self.output_dim[0]) for j in range(self.output_dim[1])]))
        plt.axis('off')
        plt.show()


# Generate sample data
data = np.random.rand(100, 2)

# Create and train the SOM
som = SOM(input_dim=(2,), output_dim=(10, 10), lr=0.1)
som.train(data, n_epochs=100)

# Get the labels and centroids
labels, centroids = som.predict(data)

# Print the labels and centroids
print(labels)
print(centroids)

