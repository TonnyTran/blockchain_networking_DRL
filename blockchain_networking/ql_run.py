from __future__ import print_function
from collections import deque
from rl.agents.tabular_q_learner import QLearner
import numpy as np
from blockchain_networking_env import BlockchainNetworkingEnv
from mempool import Mempool
import xlwt

ENV_NAME = 'BlockChain_Networking'
env = BlockchainNetworkingEnv()

def digitalizeState(observation, env=None):
    state = 0
    for index in range(0, Mempool.NB_FEE_INTERVALS):
        state += observation[index] * (Mempool.MAX_TRANSACTIONS ** index)
    for iOb in range(0, env.nb_past_observations):
        state += (observation[Mempool.NB_FEE_INTERVALS+iOb*2]
                 * (1 + observation[Mempool.NB_FEE_INTERVALS+iOb*2+1] * (env.nb_channels + 1))) ** iOb\
                 * Mempool.MAX_TRANSACTIONS ** Mempool.NB_FEE_INTERVALS
    return state

state_dim = (Mempool.MAX_TRANSACTIONS ** Mempool.NB_FEE_INTERVALS) * ((env.nb_channels + 1) * 3)**env.nb_past_observations
num_actions = env.action_space.n



nb_steps = 2000000
anneal_steps = 2000000
MAX_STEPS = 200
MAX_EPISODES = nb_steps / MAX_STEPS

version = '1.1'
q_learner = QLearner(state_dim, num_actions, anneal_steps=anneal_steps, e_vary=True)

# open workbook to store result
workbook = xlwt.Workbook()
sheet = workbook.add_sheet('QL')

episode_history = deque(maxlen=100)
for i_episode in range(MAX_EPISODES):

    # initialize
    observation = env.reset()
    state = digitalizeState(observation, env)
    action = q_learner.initializeState(state)
    episode_reward = np.zeros(4, dtype=np.float32)

    for t in range(MAX_STEPS):
        env.render()
        observation, reward, done, _ = env.step(action)

        state = digitalizeState(observation, env)

        episode_reward += reward
        action = q_learner.updateModel(state, reward[0])

        if done: break

    episode_history.append(episode_reward[0])
    mean_rewards = np.mean(episode_history)

    sheet.write(i_episode + 1, 0, str(i_episode))
    sheet.write(i_episode + 1, 1, str(episode_reward[0]))
    sheet.write(i_episode + 1, 2, str(episode_reward[1]))
    sheet.write(i_episode + 1, 3, str(episode_reward[2]))
    sheet.write(i_episode + 1, 4, str(episode_reward[3]))

    print("Episode {}".format(i_episode))
    print("Finished after {} timesteps".format(t + 1))
    print("Reward for this episode: {}".format(episode_reward[0]))
    print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
file_name = 'result_v' + version + '_QL.xls'
workbook.save('../results/' + file_name)