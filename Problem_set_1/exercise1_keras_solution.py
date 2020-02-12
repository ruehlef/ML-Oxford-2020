from __future__ import print_function  # for backwards compatibility: uses print() also in python2

from keras.models import Sequential  # feed foorward NN
from keras.layers import Dense  # need only fully connected (dense) layers
from keras import optimizers  # for gradient descent
import numpy as np

import matplotlib as mpl  # for plotting
import matplotlib.pyplot as plt  # for plotting
from mpl_toolkits.mplot3d import Axes3D  # for plotting

# Optional: Seed the random number generator for reproducibility
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)


########################################################################################################################
# (a) Implement the function f
########################################################################################################################
def f(q1, q2):
    return int(q1 * q2 < 0)


########################################################################################################################
# (b) Set up and initialize the NN
########################################################################################################################
# create the NN
my_nn = Sequential()
my_nn.add(Dense(10, activation='tanh', input_dim=2))
my_nn.add(Dense(10, activation='tanh', input_dim=10))
my_nn.add(Dense(1, activation='sigmoid', input_dim=10))


########################################################################################################################
# Define the NN hyperparameters
########################################################################################################################
learning_rate = 0.01
batch_size = 1
loss = 'binary_crossentropy'
epochs = 100


########################################################################################################################
# Specify the optimizer
########################################################################################################################
sgd = optimizers.sgd(lr=learning_rate)  # use SGD optimizer
my_nn.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
my_nn.summary()


########################################################################################################################
# Generate the training set
########################################################################################################################
x, y = [], []
for q1 in range(-5, 6):
    for q2 in range(-5,6):
        x.append([q1, q2])
        y.append(f(q1, q2))


########################################################################################################################
# (c) Train the NN
########################################################################################################################
y_hat = my_nn.fit(np.asarray(x), np.asarray(y), epochs=epochs, batch_size=batch_size, verbose=1)
print("\nTraining complete!")


########################################################################################################################
# (d) Plot the loss during training
########################################################################################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y_hat.history['loss'], 'o')  # show average losses
plt.title("Average losses per epoch")
plt.xlabel("Epochs")
plt.ylabel("Avg accuracy")
plt.yscale('log')
plt.savefig("./example_1.1_loss.pdf", dpi=300, bbox_inches='tight')
plt.close()


########################################################################################################################
# Plot prediction of NN
########################################################################################################################
y_hat = my_nn.predict(np.array(x))

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
ax.scatter([q[0] for q in x], [q[1] for q in x], [np.round(f) for f in y_hat])
ax.set_xlabel('$q_1$')
ax.set_ylabel('$q_2$')
ax.set_zlabel('y')
plt.savefig("./example_1.1_prediction_rounded.pdf", dpi=300, bbox_inches='tight')
plt.close()


########################################################################################################################
# (e) Implement with three sets
########################################################################################################################
# new labeling function. Assigns class '[1,0,0]' to 'attractive', '[0,1,0]' to 'neutral', '[0,0,1]' to 'repulsive'
def f(q1, q2):
    if q1*q2 < 0:
        return [1, 0, 0]
    elif q1*q2 == 0:
        return [0, 1, 0]
    else:
        return [0, 0, 1]

# new NN, now with three output nodes and softmax activation
my_nn = Sequential()
my_nn.add(Dense(10, activation='tanh', input_dim=2))
my_nn.add(Dense(10, activation='tanh', input_dim=10))
my_nn.add(Dense(3, activation='softmax', input_dim=10))

# same optimizer, but with categorical_crossentropy rather than binary_crossentropy loss now
sgd = optimizers.sgd(lr=learning_rate)  # use SGD optimizer
my_nn.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
my_nn.summary()

# same training set (with new function)
x, y = [], []
for q1 in range(-5, 6):
    for q2 in range(-5,6):
        x.append([q1, q2])
        y.append(f(q1, q2))

# same training call
y_hat = my_nn.fit(np.asarray(x), np.asarray(y), epochs=epochs, batch_size=batch_size, verbose=1)
print("\nTraining complete!")

# plot the loss
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y_hat.history['loss'], 'o')  # show average losses
plt.title("Average losses per epoch")
plt.xlabel("Epochs")
plt.ylabel("Avg accuracy")
plt.yscale('log')
plt.savefig("./example_1.1e_loss.pdf", dpi=300, bbox_inches='tight')
plt.close()

# plot the predictions
y_hat = my_nn.predict(np.array(x))
y_hat = y_hat.argmax(1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter([x[i][0] for i in range(len(x)) if y_hat[i] == 0], [x[i][1] for i in range(len(x)) if y_hat[i] == 0], c='r', marker='o', label='attractive')
ax.scatter([x[i][0] for i in range(len(x)) if y_hat[i] == 1], [x[i][1] for i in range(len(x)) if y_hat[i] == 1], c='g', marker='d', label='neutral')
ax.scatter([x[i][0] for i in range(len(x)) if y_hat[i] == 2], [x[i][1] for i in range(len(x)) if y_hat[i] == 2], c='b', marker='^', label='repulsive')
ax.set_xlabel('q1')
ax.set_ylabel('q2')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Result of NN classification")
plt.savefig("./example_1.1e_prediction.pdf", dpi=300, bbox_inches='tight')
plt.close()
