from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

# This prevents numpy from using multiple threads
os.environ['OMP_NUM_THREADS'] = '1'  # NOQA

import chainer
from chainer import functions as F
import gym
import gym_a3c
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl.agents import a3c
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies


class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    """The NN we will be using for the policy (softmax) and the state value (single output) estimation."""
    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.SoftmaxPolicy(model=links.MLP(ndim_obs, n_actions, hidden_sizes, nonlinearity=F.tanh))
        self.v  = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes, nonlinearity=F.tanh)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int, default=4)  # increase for more asynchronous workers
    parser.add_argument('--outdir', type=str, default='a3c_training', help='Directory path to save output files. If it does not exist, it will be created.')  # set directory to which output files will be written
    parser.add_argument('--env', type=str, default='1DIsing-A3C-v0')  # specify environment to explore

    parser.add_argument('--steps', type=int, default=1 * 10 ** 7)  # maximum number of steps before training ends
    parser.add_argument('--eval-interval', type=int, default=10**4)  # frequency at which the agent will be evaluated
    parser.add_argument('--eval-n-runs', type=int, default=10)  # number of evaluation runs per evaluation
    parser.add_argument('--arch', type=str, default='FFSoftmax', choices=('FFSoftmax'))  # NN to use for policy and state value estimates
    parser.add_argument('--t-max', type=int, default=5)  # increase for later truncation of the sum
    parser.add_argument('--beta', type=float, default=1e-2)    # increase for more exploration
    parser.add_argument('--gamma', type=float, default=0.99)    # increase for less discount of future rewards
    parser.add_argument('--lr', type=float, default=1 * 1e-4)  # decrease for slower learning rate
    parser.add_argument('--weight-decay', type=float, default=0)  # turn on to get weight decay

    parser.add_argument('--seed', type=int, default=17, help='Random seed [0, 2 ** 32)')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1e0)
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-1)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--logger-level', type=int, default=logging.ERROR)  # set to logging.DEBUG for (much more) information
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)

    # Set a random seed used in ChainerRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    misc.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor and process_idx == 0:
            env = chainerrl.wrappers.Monitor(env, args.outdir)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if args.render and process_idx == 0 and not test:
            env = chainerrl.wrappers.Render(env)
        return env

    sample_env = gym.make(args.env)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space

    model = A3CFFSoftmax(obs_space.low.size, action_space.n)

    opt = rmsprop_async.RMSpropAsync(lr=args.lr, eps=args.rmsprop_epsilon, alpha=0.99)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(40))
    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))

    agent = a3c.A3C(model, opt, t_max=args.t_max, gamma=args.gamma, beta=args.beta)
    if args.load:
        agent.load(args.load)

    if args.demo:
        env = make_env(0, True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_async(
            agent=agent,
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            profile=args.profile,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            max_episode_len=timestep_limit)


if __name__ == '__main__':
    main()
