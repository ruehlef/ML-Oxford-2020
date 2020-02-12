# Supplementary material for the Lecture "Introduction to ML for Physicists"

The syllabus of the course is available [here](./Syllabus.pdf). The repo also contains the exercises and solutions of the course. The course is based on my Physics Report [Data science applications to string theory](https://doi.org/10.1016/j.physrep.2019.09.005). If you find the code provided here useful in your research, please consider [citing it](./PhysicsReport.bib).

## Week 5
In this [exercise](./Problem_set_1/problem_set_1.pdf), we set up a Python environment for Machine learning.  We create our first feed-forward neural network that learns to classify data into two or three classes using [Keras](https://keras.io) and [PyTorch](https://pytorch.org/docs/stable/). 
<p align="center">
<img src="./Problem_set_1/example_1.1_prediction.png" width="300px"/>&nbsp;&nbsp;&nbsp;<img src="./Problem_set_1/example_1.1_loss.png" width="300px"/>
</p>

We also use CMS collider data, publicly available from [CERN opendata](http://opendata.web.cern.ch), to learn to predict the invariant mass of the Z boson in a Z &rarr; e+ e- decay. The repository for the exercise with templates and solutions is available [here](./Problem_set_1).

<p align="center">
<img src="./Problem_set_1/Invariant_mass.png" width="300px"/>&nbsp;&nbsp;&nbsp;<img src="./Problem_set_1/example_1.1e_loss.png" width="300px"/>
</p>

## Week 6
In this [exercise](./Problem_set_2/problem_set_2.pdf), we create our first convolutional neural network using PyTorch. We classify galaxies into spiral, elliptical or unknown. The data is provided by the [Galaxy Zoo](https://data.galaxyzoo.org) project. See [this publication](http://adsabs.harvard.edu/abs/2008MNRAS.389.1179L) for more details. The pictures of the galaxies themselves are provided by the [Sloan Digital Sky Survey](https://www.sdss.org). The repository for the exercise with templates and solutions is available [here](./Problem_set_2/).

NN preidction:
<table>
<tr>
<td><img src="./Problem_set_2/galaxy_spiral.png" width="200px"/></td><td><img src="./Problem_set_2/galaxy_elliptical.png" width="200px"/></td><td><img src="./Problem_set_2/galaxy_unknown.png" width="200px"/></td>
</tr><tr>
<td>spiral: 82%, elliptical: 10%, unknown: 8%</td><td>spiral: 8%, elliptical: 90%, unknown: 2%</td><td>spiral: 12%, elliptical: 10%, unknown: 78%</td>
</tr>
</table>

(Image source: [https://www.sdss.org](https://www.sdss.org))

## Week 7
In this [exercise](./Problem_set_3/problem_set_3.pdf), we demonstrate how to code an environment that can be connected via the OpenAI gym to [ChainerRL](https://github.com/chainer/chainerrl). We illustrate how the the A3C agent finds good energy configurations for the 1D Ising model by flipping spins at any of the lattice sites. The repository for the exercise with templates and solutions is available [here](./Problem_set_3/).

```
I found an optimal configuration!
↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
I started from
↑ ↑ ↓ ↓ ↑ ↑ ↑ ↑ ↑ ↓ ↓ ↑ ↓ ↑ ↑ ↓ ↑
and took the actions
[10, 2, 3, 9, 12, 15]
``` 

## Week 8
In this [exercise](./Problem_set_4/problem_set_4.pdf), we illustrate different unsupervised clustering algorithms (k-means, mean shift, DBSCAN, Birch) that were discussed in class using [scikit learn](https://scikit-learn.org/stable/).

<p align="center">
<img src="./Problem_set_4/KMeans.png" width="400px"/>&nbsp;&nbsp;&nbsp;<img src="./Problem_set_4/Birch.png" width="400px"/>
</p>

We furthermore use genetic algorithms to minimize a superpotential of an N=1 SUSY theory, which arises e.g. from string theory.

<p align="center">
<img src="./Problem_set_4/Minimum_kahler.png" width="400px"/>&nbsp;&nbsp;&nbsp;<img src="./Problem_set_4/Minimum_axions.png" width="400px"/>
</p>

 The repository for the exercise with templates and solutions is available [here](./Problem_set_4/).
