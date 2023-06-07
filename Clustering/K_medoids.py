import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs

def k_medoids(points, k):
  from scipy.spatial.distance import cdist
  from sklearn_extra.cluster import KMedoids

  # K-Medoids clustering to divide the area of broken sensors
  kmedoids = KMedoids(k, random_state=0).fit(points)
  labels = kmedoids.labels_
  medoid_indices = kmedoids.medoid_indices_

  centroids = points[medoid_indices]

  dict_dividedPoints = { i: points[labels == i] for i in range(k) }
  
  return centroids, dict_dividedPoints
