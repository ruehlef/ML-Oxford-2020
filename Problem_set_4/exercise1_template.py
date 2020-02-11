import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time  # for runtime comparisons of algorithms
from sklearn import cluster  # for clustering algorithms

# read in data
# download from http://opendata.web.cern.ch/record/545
zee_data = pd.read_csv('./Zee.csv')

# plot data
plt.scatter(zee_data.eta1 - zee_data.eta2, zee_data.phi1 - zee_data.phi2, s=10, c='b')
plt.xlabel('$\Delta\eta$')
plt.ylabel('$\Delta\phi$')
plt.title(r'$\Delta\eta$ vs $\Delta\phi$ for $Z\to e\bar e$')
plt.show()

# look at what different clustering algorithms find
dataset = [zee_data.eta1 - zee_data.eta2, zee_data.phi1 - zee_data.phi2]
x = np.asarray([[dataset[0][i], dataset[1][i]] for i in range(len(dataset[0]))])


########################################################################################################################
# (b) K means
########################################################################################################################
alg = cluster.KMeans(n_clusters=2)  # select algorithm
# implement clustering here


########################################################################################################################
# (c) Mean shift
########################################################################################################################
bandwidth = cluster.estimate_bandwidth(x, quantile=0.3)
alg = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
# implement clustering here


########################################################################################################################
# (d) DBSCAN
########################################################################################################################
alg = cluster.DBSCAN(eps=0.4)
# implement clustering here


########################################################################################################################
# (e) Birch
########################################################################################################################
alg = cluster.Birch()
# implement clustering here
