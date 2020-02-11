from __future__ import print_function  # for backwards compatibility: uses print() also in python2

from keras.models import Sequential  # feed foorward NN
from keras.layers import Dense  # need only fully connected (dense) layers
from keras import optimizers  # for gradient descent
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt  # for plotting

# Optional: Seed the random number generator for reproducibility
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)


########################################################################################################################
# (a) Implement the function for M_z
########################################################################################################################
def get_invariant_mass(pt1, pt2, eta1, eta2, phi1, phi2):
    return np.sqrt(2*pt1*pt2*(np.cosh(eta1-eta2)-np.cos(phi1-phi2)))


########################################################################################################################
# (b) Set up and initialize the NN
########################################################################################################################
# create the NN
my_nn = Sequential()
my_nn.add(Dense(100, activation='relu', input_dim=6))
my_nn.add(Dense(100, activation='relu', input_dim=100))
my_nn.add(Dense(1, activation='linear', input_dim=100))


########################################################################################################################
# (c) Define the NN hyperparameters
########################################################################################################################
learning_rate = 0.0001
batch_size = 1
loss = 'mean_squared_error'
epochs = 100


########################################################################################################################
# Specify the optimizer
########################################################################################################################
sgd = optimizers.sgd(lr=learning_rate)  # use SGD optimizer
my_nn.compile(loss=loss, optimizer=sgd, metrics=['mse'])
my_nn.summary()


########################################################################################################################
# Generate the training set
########################################################################################################################
x, y = [], []
zee_data = pd.read_csv('./Zee.csv')
inv_mass = get_invariant_mass(zee_data.pt1, zee_data.pt2, zee_data.eta1, zee_data.eta2, zee_data.phi1, zee_data.phi2)
for index, row in zee_data.iterrows():
   x.append([row['pt1'], row['pt2'], row['eta1'], row['eta2'], row['phi1'], row['phi2']])

x = np.asarray(x)
y = np.asarray(inv_mass)


########################################################################################################################
# Train the NN
########################################################################################################################
y_hat = my_nn.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1)
print("\nTraining complete!")


########################################################################################################################
# Plot the loss during training
########################################################################################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y_hat.history['loss'], 'o')  # show average losses
plt.title("Average losses per epoch")
plt.xlabel("Epochs")
plt.ylabel("Avg accuracy")
plt.yscale('log')
plt.savefig("./example_1.2c_loss.pdf", dpi=300, bbox_inches='tight')
plt.close()


########################################################################################################################
# Plot prediction of NN
########################################################################################################################
y_hat = my_nn.predict(x)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(range(len(y_hat)), [float(y_hat[i]) for i in range(len(y_hat))])
ax.set_xlabel('datapoint')
ax.set_ylabel('$\hat{y}$ [GeV]')
# plt.savefig("./example_1.2c_prediction.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(range(len(y_hat)), [float(y_hat[i])/float(y[i]) for i in range(len(y_hat))])
ax.set_xlabel('datapoint')
ax.set_ylabel('$\hat{y}/y$')
# plt.savefig("./example_1.2c_rel_error.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()


########################################################################################################################
# (d) Generate the training set
########################################################################################################################
x, y = [], []
zee_data = pd.read_csv('./Zee.csv')
inv_mass = get_invariant_mass(zee_data.pt1, zee_data.pt2, zee_data.eta1, zee_data.eta2, zee_data.phi1, zee_data.phi2)
for index, row in zee_data.iterrows():
   x.append([row['pt1']/100., row['pt2']/100., row['eta1'], row['eta2'], row['phi1'], row['phi2']])

x = np.asarray(x)
y = np.asarray(inv_mass)/100.


########################################################################################################################
# Set up and initialize the NN
########################################################################################################################
my_nn = Sequential()
my_nn.add(Dense(100, activation='relu', input_dim=6))
my_nn.add(Dense(100, activation='relu', input_dim=100))
my_nn.add(Dense(1, activation='linear', input_dim=100))
learning_rate = 0.01
batch_size = 1
loss = 'mean_squared_error'
epochs = 100
sgd = optimizers.sgd(lr=learning_rate)  # use SGD optimizer
my_nn.compile(loss=loss, optimizer=sgd, metrics=['mse'])
my_nn.summary()


########################################################################################################################
# Train the NN
########################################################################################################################
y_hat = my_nn.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1)
print("\nTraining complete!")


########################################################################################################################
# Plot the loss during training
########################################################################################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y_hat.history['loss'], 'o')  # show average losses
plt.title("Average losses per epoch")
plt.xlabel("Epochs")
plt.ylabel("Avg accuracy")
plt.yscale('log')
plt.savefig("./example_1.2d_loss.pdf", dpi=300, bbox_inches='tight')
plt.close()


########################################################################################################################
# Plot prediction of NN
########################################################################################################################
y_hat = my_nn.predict(x)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(range(len(y_hat)), [100 * float(y_hat[i]) for i in range(len(y_hat))])
ax.set_xlabel('datapoint')
ax.set_ylabel('$\hat{y}$ [GeV]')
# plt.savefig("./example_1.2d_prediction.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(range(len(y_hat)), [float(y_hat[i])/float(y[i]) for i in range(len(y_hat))])
ax.set_xlabel('datapoint')
ax.set_ylabel('$\hat{y}/y$')
# plt.savefig("./example_1.2d_rel_error.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()


########################################################################################################################
# (e) Generate the training set
########################################################################################################################
x, y = [], []
zee_data = pd.read_csv('./Zee.csv')
inv_mass = get_invariant_mass(zee_data.pt1, zee_data.pt2, zee_data.eta1, zee_data.eta2, zee_data.phi1, zee_data.phi2)
for index, row in zee_data.iterrows():
   x.append([row['pt1'] * row['pt2'] / (100*100), abs(row['eta1'] - row['eta2']), abs(row['phi1'] - row['phi2'])])

x = np.asarray(x)
y = np.asarray(inv_mass)/100.


########################################################################################################################
# Set up and initialize the NN
########################################################################################################################
my_nn = Sequential()
my_nn.add(Dense(100, activation='relu', input_dim=3))
my_nn.add(Dense(100, activation='relu', input_dim=100))
my_nn.add(Dense(1, activation='linear', input_dim=100))
learning_rate = 0.01
batch_size = 32
loss = 'mean_squared_error'
epochs = 100
sgd = optimizers.sgd(lr=learning_rate)  # use SGD optimizer
my_nn.compile(loss=loss, optimizer=sgd, metrics=['mse'])
my_nn.summary()


########################################################################################################################
# Train the NN
########################################################################################################################
y_hat = my_nn.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1)
print("\nTraining complete!")


########################################################################################################################
# Plot the loss during training
########################################################################################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y_hat.history['loss'], 'o')  # show average losses
plt.title("Average losses per epoch")
plt.xlabel("Epochs")
plt.ylabel("Avg accuracy")
plt.yscale('log')
plt.savefig("./example_1.2r_loss.pdf", dpi=300, bbox_inches='tight')
plt.close()


########################################################################################################################
# Plot prediction of NN
########################################################################################################################
y_hat = my_nn.predict(x)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(range(len(y_hat)), [100 * float(y_hat[i]) for i in range(len(y_hat))])
ax.set_xlabel('datapoint')
ax.set_ylabel('$\hat{y}$ [GeV]')
# plt.savefig("./example_1.2e_prediction.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(range(len(y_hat)), [float(y_hat[i])/float(y[i]) for i in range(len(y_hat))])
ax.set_xlabel('datapoint')
ax.set_ylabel('$\hat{y}/y$')
# plt.savefig("./example_1.2e_rel_error.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
