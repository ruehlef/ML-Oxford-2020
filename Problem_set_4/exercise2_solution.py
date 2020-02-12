import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy


def V(s, t, eta, tau):
    return 1./(16.*s*t**3) * (1 + (4*(1+4*s+4*s**2))*np.exp(-2*s) + (100*(1+4*np.pi*t+4./3.*np.pi**2*t**2))*np.exp(-2*np.pi*t) - 4*(1+2*s)*np.exp(-s)*np.cos(eta) - 20*(1+2*np.pi*t)*np.exp(-np.pi*t)*np.cos(np.pi*tau) + 40*(1+2*s+2*np.pi*t)*np.exp(-s-np.pi*t)*np.cos(eta-tau))


########################################################################################################################
# (a) Make contour plots
########################################################################################################################
s = np.arange(2, 5, 0.01)
t = np.arange(1, 2, 0.01)
ss, tt = np.meshgrid(s, t)
v = np.array([V(ss[i], tt[i], 0, 0) for i in range(len(ss))])
contour_plot = plt.contourf(ss, tt, v, 30)
plt.colorbar(contour_plot)

plt.title('Contour Plot in $s,t$ plane')
plt.xlabel('s')
plt.ylabel('t')
plt.show()

eta = np.arange(-np.pi, np.pi, 0.01)
tau = np.arange(-np.pi, np.pi, 0.01)
eeta, ttau = np.meshgrid(eta, tau)
v = np.array([V(3., 1.1, eeta[i], ttau[i]) for i in range(len(eeta))])
contour_plot = plt.contourf(eeta, ttau, v, 30)
plt.colorbar(contour_plot)

plt.title(r'Contour Plot in $\eta,\tau$ plane')
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\tau$')
plt.show()


########################################################################################################################
# (b) Implement tournament selection
########################################################################################################################
def fitness(individual):
    s, t, eta, tau = individual
    return -V(s, t, eta, tau)


def binary_tournament_select(e1, e2):
    return fitness(e1) > fitness(e2)


########################################################################################################################
# (c) Implement mutation routine
########################################################################################################################
def mutate(individual, mutation_rate=0.05):
    for i in range(len(individual)):
        if np.random.rand() <= mutation_rate:
            individual[i] += np.random.normal(0, 0.2)

    individual[0] = max(individual[0], 1)
    individual[1] = max(individual[1], 1)
    return individual


########################################################################################################################
# (d) Implement crossover routine
########################################################################################################################
def crossover(parent1, parent2):
    r1 = np.random.rand()
    r2 = np.random.rand()
    c1, c2 = np.zeros(4), np.zeros(4)
    for i in range(len(parent1)):
        c1[i] = r1 * parent1[i] + (1 - r1) * parent2[i]
        c2[i] = r2 * parent2[i] + (1 - r2) * parent1[i]

    return c1, c2


########################################################################################################################
# (e) Implement survival routine
########################################################################################################################
def survival_selection(parents, children):
    fitness_parents = np.array([fitness(e) for e in parents])
    fitness_children = np.array([fitness(e) for e in children])
    parents, children = np.asarray(parents), np.asarray(children)
    new_generation = np.append(parents[np.argsort(fitness_parents)[-70:]], children[np.argsort(fitness_children)[-30:]], 0)
    return new_generation


########################################################################################################################
# (f) Run the GA
########################################################################################################################
def get_top_five(generation):
    fitness_generation = np.array([fitness(e) for e in generation])
    generation = np.asarray(generation)
    top_five = np.array(generation[np.argsort(fitness_generation)[-5:]])
    return top_five, np.sort(fitness_generation)[-5:]


NUM_GENERATIONS = 300
NUM_INDIVIDUALS = 100
MUTATION_RATE = 0.05

generation = []
# randomly in initialized first generation
for i in range(NUM_INDIVIDUALS):
    generation.append(np.array([10*np.random.rand() + 1, 10*np.random.rand() + 1, np.pi*(np.random.rand() - 0.5), np.pi*(np.random.rand() - 0.5)]))

generations = [generation]
for g in range(NUM_GENERATIONS):
    children =[]
    for i in range(50):
        contestant1, contestant2 = generation[np.random.choice(len(generation))], generation[np.random.choice(len(generation))]
        winner1 = contestant1 if binary_tournament_select(contestant1, contestant2) else contestant2
        contestant3, contestant4 = generation[np.random.choice(len(generation))], generation[np.random.choice(len(generation))]
        winner2 = contestant3 if binary_tournament_select(contestant3, contestant4) else contestant4
        c1, c2 = crossover(winner1, winner2)
        c1, c2 = mutate(c1, MUTATION_RATE), mutate(c2, MUTATION_RATE)
        children.append(c1)
        children.append(c2)

    generation = copy.deepcopy(survival_selection(generation, children))
    generations.append(copy.deepcopy(generation))
    top_ind, top_fit = get_top_five(generation)
    print("Generation", g)
    for i in range(len(top_ind)):
        print("Individual", top_ind[i], "has fitness", top_fit[i])
    print("########################################################################################################################")

top_ind, top_fit = get_top_five(generation)
winner = top_ind[0]
print("########################################################################################################################")
print("Best solution:")
print("[s, t, eta, tau] = [{:3f}, {:3f}, {:3f}, {:3f}]".format(winner[0], winner[1], winner[2], winner[3]))
print("########################################################################################################################")

s = np.arange(2, 5, 0.01)
t = np.arange(1, 2, 0.01)
ss, tt = np.meshgrid(s, t)
v = np.array([V(ss[i], tt[i], winner[2], winner[3]) for i in range(len(ss))])
contour_plot = plt.contourf(ss, tt, v, 30)
plt.colorbar(contour_plot)
plt.scatter(winner[0], winner[1], c='r', marker='o')

plt.title('Contour Plot in $s,t$ plane')
plt.xlabel('s')
plt.ylabel('t')
plt.show()

eta = np.arange(-np.pi, np.pi, 0.01)
tau = np.arange(-np.pi, np.pi, 0.01)
eeta, ttau = np.meshgrid(eta, tau)
v = np.array([V(winner[0], winner[1], eeta[i], ttau[i]) for i in range(len(eeta))])
contour_plot = plt.contourf(eeta, ttau, v, 30)
plt.colorbar(contour_plot)
plt.scatter(winner[2], winner[3], c='r', marker='o')

plt.title(r'Contour Plot in $\eta,\tau$ plane')
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\tau$')
plt.show()
