from gym.envs.registration import register

###### 1D Ising environment
register(
    id='1DIsing-A3C-v0',
    entry_point='gym_a3c.Ising1D_env:Ising1DEnv',
    max_episode_steps=50
)