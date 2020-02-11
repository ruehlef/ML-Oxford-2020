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
    # your implementation goes here


########################################################################################################################
# (b) Set up and initialize the NN
########################################################################################################################
# create the NN
my_nn = Sequential()
my_nn.add(Dense(10, activation='tanh', input_dim=2))
# add the other layers here


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
