import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class Ising1DEnv(gym.Env):

    ####################################################################################################################
    # (b) Implement __init__
    ####################################################################################################################
    def __init__(self):
        # define the following members
        self.state =
        self.action_space =   # you might want to use spaces.Discrete(number_of_actions)
        self.observation_space =  # you might want to use spaces.Box(-1, 1, [size_of_lattice], dtype=np.float32)

    ####################################################################################################################
    # (c) Implement reward function
    ####################################################################################################################
    def reward(self):
        # define the reward function based on the current state as stored in self.state. It should return the reward and whether the state is terminal
        reward =
        is_terminal_state =
        return reward, is_terminal_state

    ####################################################################################################################
    # (d) Implement reward function
    ####################################################################################################################
    def step(self, action):
        self.state =   # change state based on action
        my_reward, is_terminal_state = self.reward()  # gets reward and terminal state info for new state

        return np.array(self.state), my_reward, is_terminal_state, {}

    ####################################################################################################################
    # (e) Implement reset function
    ####################################################################################################################
    def reset(self):
        self.state =   # reset state to an initial state
        return np.array(self.state)

    ####################################################################################################################
    # Other functions
    ####################################################################################################################

    # we don't use seeding, but it's an abstract class method, so we get a warning if we don't implement it
    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets checked as an int elsewhere, so we need to keep it below 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
