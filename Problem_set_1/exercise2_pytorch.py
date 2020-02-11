#################################################################################################
# Keras implementation of the simple NN that classifies bundle stability (cf. Section 2.1)      #
#################################################################################################
from __future__ import print_function  # for backwards compatibility: uses print() also in python2

import torch
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
my_nn = torch.nn.Sequential(
    # implement the NN here as in Exercise 1
)


########################################################################################################################
# (c) Define the NN hyperparameters
########################################################################################################################
learning_rate = 0.0001
epochs = 100


########################################################################################################################
# Specify the optimizer
########################################################################################################################
sgd = torch.optim.SGD(my_nn.parameters(), lr=learning_rate)  # use SGD optimizer
criterion = torch.nn.MSELoss()


########################################################################################################################
# Generate the training set
########################################################################################################################
x, y = [], []
zee_data = pd.read_csv('./Zee.csv')
inv_mass = get_invariant_mass(zee_data.pt1, zee_data.pt2, zee_data.eta1, zee_data.eta2, zee_data.phi1, zee_data.phi2)
for index, row in zee_data.iterrows():
   x.append([row['pt1'], row['pt2'], row['eta1'], row['eta2'], row['phi1'], row['phi2']])

x = np.asarray(x)
y = np.asarray([[y] for y in inv_mass])
x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


########################################################################################################################
# Train the NN
########################################################################################################################
avg_losses = []
for epoch in range(epochs):
    avg_loss = 0
    for i in range(len(x)):
        sgd.zero_grad()
        y_hat = my_nn(x[i])
        loss = criterion(y_hat, y[i])
        loss.backward()
        avg_loss += loss.data
        sgd.step()

    avg_losses.append(avg_loss)
    print("Epoch {:3d}: MSE Loss {:5f}".format(epoch, avg_loss))

print("\nTraining complete!")


########################################################################################################################
# Plot the loss during training
########################################################################################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(avg_losses, 'o')  # show average losses
plt.title("Average losses per epoch")
plt.xlabel("Epochs")
plt.ylabel("Avg accuracy")
plt.yscale('log')
plt.savefig("./example_1.2c_loss.pdf", dpi=300, bbox_inches='tight')
plt.close()


########################################################################################################################
# Plot prediction of NN
########################################################################################################################
y_hat = my_nn(x)
x, y_hat = x.detach().numpy(), y_hat.detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(range(len(y_hat)), [float(y_hat[i]) for i in range(len(y_hat))])
ax.set_xlabel('datapoint')
ax.set_ylabel('$\hat{y}$')
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
# (d) Generate the training set and train the NN
########################################################################################################################
x, y = [], []
# Generate the training set here

########################################################################################################################
# Set up and initialize the NN
########################################################################################################################
my_nn = torch.nn.Sequential(
    # your implementation goes here
)

learning_rate = 0.01
epochs = 100
sgd = torch.optim.SGD(my_nn.parameters(), lr=learning_rate)  # use SGD optimizer
criterion = torch.nn.MSELoss()


########################################################################################################################
# Train the NN
########################################################################################################################
avg_losses = []
for epoch in range(epochs):
    avg_loss = 0
    for i in range(len(x)):
        sgd.zero_grad()
        y_hat = my_nn(x[i])
        loss = criterion(y_hat, y[i])
        loss.backward()
        avg_loss += loss.data
        sgd.step()

    avg_losses.append(avg_loss)
    print("Epoch {:3d}: MSE Loss {:5f}".format(epoch, avg_loss))

print("\nTraining complete!")


########################################################################################################################
# Plot the loss during training
########################################################################################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(avg_losses, 'o')  # show average losses
plt.title("Average losses per epoch")
plt.xlabel("Epochs")
plt.ylabel("Avg accuracy")
plt.yscale('log')
plt.savefig("./example_1.2d_loss.pdf", dpi=300, bbox_inches='tight')
plt.close()


########################################################################################################################
# Plot prediction of NN
########################################################################################################################
y_hat = my_nn(x)
x, y_hat = x.detach().numpy(), y_hat.detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(range(len(y_hat)), [float(y_hat[i]) for i in range(len(y_hat))])
ax.set_xlabel('datapoint')
ax.set_ylabel('$\hat{y}$')
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
# (e) Generate the training set and train the NN
########################################################################################################################
x, y = [], []
# Generate the training set here


########################################################################################################################
# Set up and initialize the NN
########################################################################################################################
my_nn = torch.nn.Sequential(
    # your implementation goes here
)

learning_rate = 0.01
epochs = 100
sgd = torch.optim.SGD(my_nn.parameters(), lr=learning_rate)  # use SGD optimizer
criterion = torch.nn.MSELoss()


########################################################################################################################
# Train the NN
########################################################################################################################
avg_losses = []
for epoch in range(epochs):
    avg_loss = 0
    for i in range(len(x)):
        sgd.zero_grad()
        y_hat = my_nn(x[i])
        loss = criterion(y_hat, y[i])
        loss.backward()
        avg_loss += loss.data
        sgd.step()

    avg_losses.append(avg_loss)
    print("Epoch {:3d}: MSE Loss {:5f}".format(epoch, avg_loss))

print("\nTraining complete!")


########################################################################################################################
# Plot the loss during training
########################################################################################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(avg_losses, 'o')  # show average losses
plt.title("Average losses per epoch")
plt.xlabel("Epochs")
plt.ylabel("Avg accuracy")
plt.yscale('log')
plt.savefig("./example_1.2e_loss.pdf", dpi=300, bbox_inches='tight')
plt.close()


########################################################################################################################
# Plot prediction of NN
########################################################################################################################
y_hat = my_nn(x)
x, y_hat = x.detach().numpy(), y_hat.detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(range(len(y_hat)), [float(y_hat[i]) for i in range(len(y_hat))])
ax.set_xlabel('datapoint')
ax.set_ylabel('$\hat{y}$')
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
