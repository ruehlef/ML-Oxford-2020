from __future__ import print_function  # for backwards compatibility: uses print() also in python2

import torch

import matplotlib.pyplot as plt  # for plotting
from mpl_toolkits.mplot3d import Axes3D  # for plotting


########################################################################################################################
# (a) Implement the function f
########################################################################################################################
def f(q1, q2):
    # your implementation goes here


########################################################################################################################
# (b) Set up and initialize the NN
########################################################################################################################
# create the NN
my_nn = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.Tanh()
    # add the other layers here
)


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

    avg_losses.append(avg_loss / float(len(x)))

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
ax.set_xlabel('q1')
ax.set_ylabel('q2')
ax.set_zlabel('y')
plt.savefig("./example_1.1_prediction.pdf", dpi=300, bbox_inches='tight')
plt.close()


########################################################################################################################
# Visualize with decision boundary 0.5
########################################################################################################################
# implement the code to plot the rounded output here
plt.savefig("./example_1.1_prediction_rounded.pdf", dpi=300, bbox_inches='tight')
plt.close()


########################################################################################################################
# (e) Implement with three sets
########################################################################################################################
# your code goes here
