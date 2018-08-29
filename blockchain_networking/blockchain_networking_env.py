import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from action_space import ActionSpace
from channel_space import ChannelSpace
from state_space import StateSpace
from mempool import Mempool, Transaction, Block

from gym.spaces import Discrete

class BlockchainNetworkingEnv(gym.Env):
    SUCCESS_REWARD = 5
    LATE_PROB = 1
    MAX_ATTACK = 0.1

    def __init__(self):
        # Channel parameters
        self.nb_channels = 4
        self.idleChannel = 1
        self.prob_switching = 0.9
        self.channelObservation = None
        self.prob_late = BlockchainNetworkingEnv.LATE_PROB
        self.cost_channels = [0.1, 0.1, 0.1, 0.1]

        # Blockchain parameters
        self.mempool = Mempool()
        self.userTransaction = Transaction()
        self.lastBlock = Block()
        self.hashRate = None
        self.doubleSpendSuccess = None

        # System parameters
        self.nb_past_observations = 4

        self.state_size = Mempool.NB_FEE_INTERVALS + 2*self.nb_past_observations


        self.action_space = ActionSpace(self.nb_channels + 1)
        self.observation_space = StateSpace((Discrete(Mempool.MAX_FEE), Discrete(Mempool.MAX_FEE),
                                             Discrete(Mempool.MAX_FEE), Discrete(Mempool.MAX_FEE),
                                             Discrete(Mempool.MAX_FEE), Discrete(Mempool.MAX_FEE),
                                             Discrete(Mempool.MAX_FEE), Discrete(Mempool.MAX_FEE),
                                             Discrete(Mempool.MAX_FEE), Discrete(Mempool.MAX_FEE),
                                             ActionSpace(self.nb_channels + 1), ChannelSpace(),
                                             ActionSpace(self.nb_channels + 1), ChannelSpace(),
                                             ActionSpace(self.nb_channels + 1), ChannelSpace(),
                                             ActionSpace(self.nb_channels + 1), ChannelSpace()))
        # reward define
        self.totalReward = 0
        self.successReward = 0
        self.channelCost = 0
        self.transactionFee = 0
        self.cost = 0

        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # reset the rewards
        self.totalReward = 0
        self.successReward = 0
        self.channelCost = 0
        self.transactionFee = 0
        self.prob_late = None
        self.attacked = False

        state = list(self.state)
        # 1. User's transaction initialization
        self.userTransaction = Transaction()
        if (len(self.lastBlock.blockTransaction) != 0):
            self.userTransaction.estimateFee(self.lastBlock)

        # 2. The channel state changes - single idle channel, round robin switching
        if (np.random.rand() < self.prob_switching):
            self.idleChannel = (self.idleChannel + 1) % self.nb_channels
            # print(self.idleChannel)

        # 3. Mempool updates - some new transactions come
        self.mempool.generateNewTransactions()

        # if user does not submit transaction
        if (action == 0):
            self.totalReward = 0
            self.channelObservation = 2
            # miners mine a block
            self.lastBlock.mineBlock(self.mempool)
        # if user submits transaction
        else:
            self.channelCost = self.cost_channels[action-1]
            # in case, channel is idle
            if((action-1) == self.idleChannel):
                self.prob_late = 0
                self.channelObservation = 1
            # if channel is busy, transaction can be late of mining process
            else:
                self.prob_late = BlockchainNetworkingEnv.LATE_PROB
                self.channelObservation = 0

            # if the transaction comes late
            if(np.random.rand() < self.prob_late):
                # mining process occurs before user's transaction is added
                # 4. Miners start mining process, transactions which are included in Block will be removed from mempool
                self.lastBlock.mineBlock(self.mempool)
                self.mempool.listTransactions.append(self.userTransaction)
                self.transactionFee = self.userTransaction.transactionFee
            else:
                self.mempool.listTransactions.append(self.userTransaction)
                # 4. Miners start mining process, transactions which are included in Block will be removed from mempool
                self.lastBlock.mineBlock(self.mempool)
                self.transactionFee = self.userTransaction.transactionFee
                # 5. Attack process
                self.hashRate = np.random.uniform(0, BlockchainNetworkingEnv.MAX_ATTACK)
                self.doubleSpendSuccess = 2 * self.hashRate
                if(np.random.rand() < self.doubleSpendSuccess):
                    self.attacked = True

                # if user's transaction is successfully added inti the block -> reward=2
                if (self.userTransaction in self.lastBlock.blockTransaction and not self.attacked):
                    self.successReward = BlockchainNetworkingEnv.SUCCESS_REWARD

        self.totalReward = self.successReward - self.channelCost - self.transactionFee
        self.cost = self.channelCost + self.transactionFee

        # 6. determine new state
        self.mempool.updateMempoolState()
        for index in range(0, Mempool.NB_FEE_INTERVALS):
            state[index] = self.mempool.mempoolState[index]
        state.insert(Mempool.NB_FEE_INTERVALS, action)
        state.insert(Mempool.NB_FEE_INTERVALS+1, self.channelObservation)
        state.pop()
        state.pop()
        self.state = tuple(state)
        done = False

        # print(np.array(self.state), [self.totalReward, self.cost], done, {})
        return np.array(self.state), [self.totalReward, self.channelCost, self.transactionFee, self.cost], done, {}

    def reset(self):
        self.state = []
        self.mempool.resetMempool()
        self.idleChannel = 1
        for index in range(0, len(self.mempool.mempoolState)):
            self.state.append(self.mempool.mempoolState[index])
        for obs_index in range(0, self.nb_past_observations):
            self.state.append(0)
            self.state.append(2)
        print(self.state)
        self.steps_beyond_done = None
        return np.array(self.state)

    def updateObservation(self):
        return

    def render(self, mode='human', close=False):
       return

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise NotImplementedError()

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

# env = BlockchainNetworkingEnv()
# env.reset()
# for index in range(0, 50):
#     env.step(np.random.randint(0, env.nb_channels))