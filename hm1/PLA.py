import numpy as np
import pandas as pd

class PLA(object):
    def __init__(self,random=False,eta = 1):
        self.eta = eta
        self.dimension = 0
        self.num = 0
        self.random = random

    def data_preprocessing(self, path):
        data = pd.read_table(path,sep = '\s',header=None, engine='python')
        if self.random:
            data = data.sample(frac=1).reset_index(drop=True)
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1].values 
        y_train = y.reshape((-1,1))
        X_train = np.hstack((np.ones((X.shape[0],1),dtype=int),X))
        self.num = X_train.shape[0]
        self.dimension = X_train.shape[1]
        return X_train,y_train
    
    def iteration_count(self, path):
        count = 0
        X_train, y_train = self.data_preprocessing(path)
        W = np.zeros((self.dimension, 1))
        # loop until all x are classified right
        while True:
            flag = 0
            for i in range(self.num):
                # transform
                hypothesis = np.dot(X_train[i], W)*y_train[i]
                if hypothesis <= 0:
                    W += self.eta*y_train[i] * X_train[i].reshape((-1, 1))
                    count += 1
                    flag = 1
            if flag == 0:
                break
        return count



if __name__ == '__main__':
    sum = 0
    for i in range(2000): 
        perceptron = PLA(random=True,eta=0.5)
        sum +=perceptron.iteration_count('E:/programming/python/ML_NTU/机器学习基石/hw1_15_train.dat')
    print(sum/2000)

