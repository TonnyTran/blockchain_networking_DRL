import gym
import numpy as np
import math


# Mempool stores all unconfirmed transactions.
# Mempool is public for all people
class Mempool():
    DEFAUT_FEE = 0.05
    MAX_FEE = 2.0
    NB_FEE_INTERVALS = 3
    MAX_TRANSACTIONS = 10
    def __init__(self):
        self.listTransactions = []
        self.fee_interval = Mempool.MAX_FEE / Mempool.NB_FEE_INTERVALS
        self.mempoolState = np.zeros((Mempool.NB_FEE_INTERVALS,), dtype=int)

    def resetMempool(self):
        self.listTransactions = []
        for index in range(0, Mempool.MAX_TRANSACTIONS-1):
            if(np.random.rand() < 0.6):
                self.listTransactions.append(Transaction())
        self.updateMempoolState()

    def updateMempoolState(self):
        self.mempoolState = np.zeros((Mempool.NB_FEE_INTERVALS,), dtype=int)
        for transaction in self.listTransactions:
            category = int(math.floor(transaction.transactionFee / self.fee_interval))
            self.mempoolState[category] += 1

    def sortMempool(self):
        self.listTransactions.sort(key=lambda transaction: transaction.transactionFee, reverse=True)

    def generateNewTransactions(self):
        nb_new_transactions = min(np.random.randint(0, 4), Mempool.MAX_TRANSACTIONS-1-len(self.listTransactions))
        for index in range(0, nb_new_transactions):
            self.listTransactions.append(Transaction())

# define transaction
class Transaction():
    def __init__(self):
        self.transactionFee = np.random.uniform(Mempool.DEFAUT_FEE, Mempool.MAX_FEE)

    def estimateFee(self, lastBlock):
        totalFee = 0
        for transaction in lastBlock.blockTransaction:
            totalFee += transaction.transactionFee
        self.transactionFee = totalFee / len(lastBlock.blockTransaction)

class Block():
    def __init__(self):
        self.blockTransaction = []

    def mineBlock(self, mempool):
        self.blockTransaction=[]
        mempool.sortMempool()
        blockSize = np.random.randint(1, 3)
        for index in range(0, min(blockSize,len(mempool.listTransactions))):
            self.blockTransaction.append(mempool.listTransactions[0])
            del mempool.listTransactions[0]


# #
# mempool = Mempool()
# mempool.resetMempool()
# mempool.sortMempool()
# for transaction in mempool.listTransactions:
#     print (transaction.transactionFee)
# # mempool.updateMempoolState()
# lastBlock = Block()
# currentTransaction = Transaction()
# for index in range(0, 100):
#     lastBlock.mineBlock(mempool)
#     for transaction in lastBlock.blockTransaction:
#         print (transaction.transactionFee)
#     if currentTransaction in lastBlock.blockTransaction:
#         print("Yesssss")
#     else:
#         print("nooooo")
#     currentTransaction = Transaction()
#     currentTransaction.estimateFee(lastBlock)
#     print(currentTransaction.transactionFee)
#     mempool.generateNewTransactions()
#     mempool.listTransactions.append(currentTransaction)
#     print(len(mempool.listTransactions))
#     mempool.updateMempoolState()
#     print(mempool.mempoolState)
# print(mempool.mempoolState)