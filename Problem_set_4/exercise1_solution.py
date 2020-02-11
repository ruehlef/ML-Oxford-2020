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
plt.suptitle("KMeans", size=24)

alg = cluster.KMeans(n_clusters=2)  # select algorithm

t0 = time.time()
alg.fit(x)
t1 = time.time()
y_pred = alg.labels_.astype(np.int)

# original data
plt.subplot(1, 2, 1)
plt.scatter(zee_data.eta1 - zee_data.eta2, zee_data.phi1 - zee_data.phi2, s=10, c='b')

# clustered data
plt.subplot(1, 2, 2)
colors = np.array(['green', 'blue', 'red'])  # add red color for outliers (if any)
plt.scatter(x[:, 0], x[:, 1], s=10, color=colors[y_pred])

# show cluster centers in plot and print their coordinates to stdout
centers = alg.cluster_centers_
if len(centers) != 0:
    plt.scatter(centers[:, 0], centers[:, 1], s=100, marker='X', alpha=1, linewidth=1.5, edgecolors='black')
    print(centers)

# print timing
plt.text(.95, .075, ('{:1.4f}s'.format(t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=12, horizontalalignment='right', backgroundcolor='white')
#
plt.show()
plt.close()


########################################################################################################################
# (c) Mean shift
########################################################################################################################
plt.suptitle("Mean shift", size=24)

bandwidth = cluster.estimate_bandwidth(x, quantile=0.3)
alg = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
#
t0 = time.time()
alg.fit(x)
t1 = time.time()
y_pred = alg.labels_.astype(np.int)

# original data
plt.subplot(1, 2, 1)
plt.scatter(zee_data.eta1 - zee_data.eta2, zee_data.phi1 - zee_data.phi2, s=10, c='b')

# clustered data
plt.subplot(1, 2, 2)
colors = np.array(['green', 'blue', 'red'])  # add red color for outliers (if any)
plt.scatter(x[:, 0], x[:, 1], s=10, color=colors[y_pred])

# show cluster centers in plot and print their coordinates to stdout
centers = alg.cluster_centers_
if len(centers) != 0:
    plt.scatter(centers[:, 0], centers[:, 1], s=100, marker='X', alpha=1, linewidth=1.5, edgecolors='black')
    print(centers)

# print timing
plt.text(.95, .075, ('{:1.4f}s'.format(t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=12, horizontalalignment='right', backgroundcolor='white')

plt.show()
plt.close()


########################################################################################################################
# (d) DBSCAN
########################################################################################################################
plt.suptitle("DBSCAN", size=24)

alg = cluster.DBSCAN(eps=0.4)

t0 = time.time()
alg.fit(x)
t1 = time.time()
y_pred = alg.labels_.astype(np.int)

# original data
plt.subplot(1, 2, 1)
plt.scatter(zee_data.eta1 - zee_data.eta2, zee_data.phi1 - zee_data.phi2, s=10, c='b')

# clustered data
plt.subplot(1, 2, 2)
colors = np.array(['green', 'blue', 'red'])  # third color for outliers
plt.scatter(x[:, 0], x[:, 1], s=10, color=colors[y_pred])

# print timing
plt.text(.95, .075, ('{:1.4f}s'.format(t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=12, horizontalalignment='right', backgroundcolor='white')

plt.show()
plt.close()


########################################################################################################################
# (e) Birch
########################################################################################################################
plt.suptitle("Birch", size=24)

alg = cluster.Birch()

t0 = time.time()
alg.fit(x)
t1 = time.time()
y_pred = alg.labels_.astype(np.int)

# original data
plt.subplot(1, 2, 1)
plt.scatter(zee_data.eta1 - zee_data.eta2, zee_data.phi1 - zee_data.phi2, s=10, c='b')

# clustered data
plt.subplot(1, 2, 2)
colors = np.array(['green', 'blue', 'red'])  # add red color for outliers (if any)
plt.scatter(x[:, 0], x[:, 1], s=10, color=colors[y_pred])

# print timing
plt.text(.95, .075, ('{:1.4f}s'.format(t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=12, horizontalalignment='right', backgroundcolor='white')

plt.show()
plt.close()
