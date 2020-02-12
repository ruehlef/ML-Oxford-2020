from __future__ import print_function  # for backwards compatibility: uses print() also in python2

import torch

import matplotlib.pyplot as plt  # for plotting
from mpl_toolkits.mplot3d import Axes3D  # for plotting


########################################################################################################################
# (a) Implement the function f
########################################################################################################################
def f(q1, q2):
    return int(q1 * q2 > 0)


########################################################################################################################
# (b) Set up and initialize the NN
########################################################################################################################
# create the NN
my_nn = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.Tanh(),
    torch.nn.Linear(10, 10),
    torch.nn.Tanh(),
    torch.nn.Linear(10, 1),
    torch.nn.Sigmoid()
)

# Optional: print summary of the NN using the torchsummary package.
# Install with pip install torchsummary
# import torchsummary
# torchsummary.summary(my_nn, (2, ), batch_size=0, device='cpu')


########################################################################################################################
# Define the NN hyperparameters
########################################################################################################################
learning_rate = 0.01
epochs = 100


########################################################################################################################
# Specify the optimizer
########################################################################################################################
sgd = torch.optim.SGD(my_nn.parameters(), lr=learning_rate)  # use SGD optimizer
criterion = torch.nn.BCELoss()


########################################################################################################################
# Generate the training set
########################################################################################################################
x, y = [], []
for q1 in range(-5, 6):
    for q2 in range(-5,6):
        x.append([q1, q2])
        y.append([f(q1, q2)])

x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


########################################################################################################################
# (c) Train the NN
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

print("\nTraining complete!")


########################################################################################################################
# (d) Plot the loss during training
########################################################################################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(avg_losses, 'o')  # show average losses
plt.title("Average losses per epoch")
plt.xlabel("Epochs")
plt.ylabel("Avg accuracy")
plt.yscale('log')
plt.savefig("./example_1.1_loss.pdf", dpi=300, bbox_inches='tight')
plt.close()


########################################################################################################################
# Plot prediction of NN
########################################################################################################################
y_hat = my_nn(x)
x, y_hat = x.detach().numpy(), y_hat.detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([q[0] for q in x], [q[1] for q in x], [f for f in y_hat])
ax.set_xlabel('$q_1$')
ax.set_ylabel('$q_2$')
ax.set_zlabel('y')
plt.savefig("./example_1.1_prediction.pdf", dpi=300, bbox_inches='tight')
plt.close()


########################################################################################################################
# Visualize with decision boundary 0.5
########################################################################################################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([q[0] for q in x], [q[1] for q in x], [f.round() for f in y_hat])
ax.set_xlabel('$q_1$')
ax.set_ylabel('$q_2$')
ax.set_zlabel('y')
plt.savefig("./example_1.1_prediction_rounded.pdf", dpi=300, bbox_inches='tight')
plt.close()


########################################################################################################################
# (e) Implement with three sets
########################################################################################################################
# new labeling function. Assigns class '0' to 'attractive', '1' to 'neutral', '2' to 'repulsive'
def f(q1, q2):
    if q1*q2 < 0:
        return 0
    elif q1*q2 == 0:
        return 1
    else:
        return 2

# new NN, now with three output nodes and softmax activation
my_nn = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.Tanh(),
    torch.nn.Linear(10, 10),
    torch.nn.Tanh(),
    torch.nn.Linear(10, 3)
    # note that torch applies softmax automatically when using torch.nn.CrossEntropyLoss
)
# Optional
# import torchsummary
# torchsummary.summary(my_nn, (2, ), batch_size=0, device='cpu')

# same optimizer, but with categorical_crossentropy rather than binary_crossentropy loss now
sgd = torch.optim.SGD(my_nn.parameters(), lr=learning_rate)  # use SGD optimizer
criterion = torch.nn.CrossEntropyLoss()

# same training set (with new function)
x, y = [], []
for q1 in range(-5, 6):
    for q2 in range(-5,6):
        x.append([q1, q2])
        y.append([f(q1, q2)])

x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# same training call
avg_losses = []
for epoch in range(epochs):
    avg_loss = 0
    for i in range(len(x)):
        sgd.zero_grad()
        y_hat = my_nn(x[i])
        loss = criterion(y_hat.unsqueeze(0), y[i])
        loss.backward()
        avg_loss += loss.data
        sgd.step()
    avg_losses.append(avg_loss)
print("\nTraining complete!")

# plot the loss
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(avg_losses, 'o')  # show average losses
plt.title("Average losses per epoch")
plt.xlabel("Epochs")
plt.ylabel("Avg accuracy")
plt.yscale('log')
plt.savefig("./example_1.1e_loss.pdf", dpi=300, bbox_inches='tight')
plt.close()

# plot the predictions
y_hat = my_nn(x).detach().numpy()
y_hat = y_hat.argmax(1)
x = x.detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter([x[i][0] for i in range(len(x)) if y_hat[i] == 0], [x[i][1] for i in range(len(x)) if y_hat[i] == 0], c='r', marker='o', label='attractive')
ax.scatter([x[i][0] for i in range(len(x)) if y_hat[i] == 1], [x[i][1] for i in range(len(x)) if y_hat[i] == 1], c='g', marker='d', label='neutral')
ax.scatter([x[i][0] for i in range(len(x)) if y_hat[i] == 2], [x[i][1] for i in range(len(x)) if y_hat[i] == 2], c='b', marker='^', label='repulsive')
ax.set_xlabel('$q_1$')
ax.set_ylabel('$q_2$')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Result of NN classification")
plt.savefig("./example_1.1e_prediction.pdf", dpi=300, bbox_inches='tight')
plt.close()
