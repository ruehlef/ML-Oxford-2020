import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class Ising1DEnv(gym.Env):

    def __init__(self):
        # physics parameters 
        self.lattice_size = 20
        
        # external magnetic field
        self.h = [0 for _ in range(self.lattice_size)]
        # h = random.rand(lattice_size)
        
        # interaction strength
        self.J = 1  # ferromagnetic
        # J = -1 # anti-ferromagnetic
        
        # optimal configuration energy
        self.optimum = 0
        if self.J == 1:
            self.optimum = self.get_energy([1 for _ in range(self.lattice_size)])  # assumes h=0
        if self.J == -1:
            self.optimum = self.get_energy([(-1)**i for i in range(self.lattice_size)])  # assumes h=0
        
        # initial state
        self.start_config = np.random.choice([-1, 1], self.lattice_size)  # -1: spin down, 1: spin up
        self.state = np.copy(self.start_config)
        self.global_t = 0

        # rewards and punishments
        self.won_reward = 10000
        self.move_punishment = 0.5

        # states and actions
        self.num_actions = self.lattice_size  # one for each lattice site
        self.action_space = spaces.Discrete(self.num_actions)  # linearize action space into a number 0 to chain_length
        self.observation_space = spaces.Box(-1, 1, [self.lattice_size], dtype=np.float32)

    def state_to_str(self, state):
        res = ""
        for s in state:
            if s == -1:
                res += u'\u2193'
            elif s == 1:
                res += u'\u2191'
            else:
                res += "*"
            res += " "
        return res

    def step(self, action):
        # Carry out move
        self.state[action] *= -1
        my_rew, am_i_done = self.reward()

        return np.array(self.state), my_rew, am_i_done, {}

    def reset(self):
        # create random start state
        self.state = self.start_config
        return np.array(self.state)

    # we don't use the seeding, but it's an abstract class method, so we get a warning if we don't implement it
    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets checked as an int elsewhere, so we need to keep it below 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def get_energy(self, state):
        energy = 0
        for i in range(self.lattice_size):
            energy += self.h[i]*state[i]
            if i == 0:
                energy += self.J*state[i]*state[i+1]
            elif i == self.lattice_size - 1:
                energy += self.J*state[i]*state[i-1]
            else:
                energy += self.J*(state[i]*state[i+1] + state[i]*state[i-1])
        return energy
    
    def reward(self):
        reward = self.get_energy(self.state)
        done = reward == self.optimum

        if done:
            reward += self.won_reward
            print("#######################################################################################")
            print("I found an optimal configuration!")
            print(self.state_to_str(self.state))
            print("#######################################################################################")

        reward -= self.move_punishment
        return reward, done
            
    def set_global_t(self, global_t):
        self.global_t = global_t
        if global_t > 0 and global_t % 1000 == 0:
            print(self.state_to_str(self.state), self.get_energy(self.state))

        # print stats (only once)
        if self.global_t == 1:
            print("#######################################################################################")
            print("Running with lattice size", self.lattice_size, "and J =", self.J)
            print("The best configuration energy achievable is", self.optimum)
            print("Start configuration:")
            print(self.state_to_str(self.start_config), self.get_energy(self.start_config))
            print("External magnetic field:")
            print(self.state_to_str(self.h))
            print("#######################################################################################")

    def set_agent(self, agent):
        pass