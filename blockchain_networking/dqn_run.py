import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, MaxBoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from blockchain_networking_env import BlockchainNetworkingEnv

ENV_NAME = 'BlockChain_Networking'

# Get the environment and extract the number of actions.
env = BlockchainNetworkingEnv()
np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.nb_actions

# Next, we build a very simple model.

model = Sequential()
model.add(Flatten(input_shape=(1, env.state_size)))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()

version = "1.1"
nb_steps = 2000000
nb_max_episode_steps = 200
anneal_steps = 400000
# processor = BlockchainProcessor()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, processor=None, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy, vary_eps=True, anneal_steps=anneal_steps)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=nb_steps, visualize=True, verbose=2, log_interval=1000, nb_max_episode_steps=nb_max_episode_steps, version=version)

# After training is done, we save the final weights.
dqn.save_weights('../save_weight/dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=0, visualize=True, nb_max_episode_steps=nb_max_episode_steps)
