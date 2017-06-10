import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler



spectral = cluster.SpectralClustering(n_clusters=2,
                                        eigen_solver='arpack',
                                        affinity="nearest_neighbors")

spectral.fit(X)