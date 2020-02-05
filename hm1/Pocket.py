import numpy as np
import pandas as pd

class Pocket(object):
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
    
    def iteration(self, path, tolerance=50):     
        count = 0
        X_train, y_train = self.data_preprocessing(path)
        W = np.zeros((self.dimension, 1))
        g = np.zeros((self.dimension,1))
        best_error = self.num
        for i in range(self.num):
            hypothesis = np.dot(X_train[i],W)*y_train[i]
            if hypothesis <=0:
                W = W + self.eta*y_train[i]*X_train[i].reshape((-1,1))
                count += 1
                temp_num = 0
                #check if it is the best
                for j in range(self.num):
                    if np.dot(X_train[j],W)*y_train[j] <=0:
                        temp_num +=1
                if temp_num <= best_error:
                    best_error = temp_num
                    g = W.copy()                          
                if count == tolerance:
                    break

        return g

    def test(self,path,W):
        self.random = False
        X_test,y_test = self.data_preprocessing(path)
        error = 0
        for i in range(X_test.shape[0]):
            if np.dot(X_test[i],W)*y_test[i]<=0:
                error += 1
        return error/X_test.shape[0]

if __name__ == '__main__':
    train_set = 'E:/programming/python/ML_NTU/机器学习基石/hw1_18_train.dat'
    test_set = 'E:/programming/python/ML_NTU/机器学习基石/hw1_18_test.dat'
    pocket = Pocket(random=True,eta=0.5)
    total_error = 0
    for test in range(2000): 
        W = pocket.iteration(train_set,tolerance=100)
        total_error += pocket.test(test_set,W)
    print(total_error/2000)
