import numpy as np
import pandas as pd

def data_preprocessing(path):
    data = pd.read_table(path,sep = '\s',header=None, engine='python')
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    y_train = y.reshape((-1,1))
    X_train = np.hstack((np.ones((X.shape[0],1),dtype=int),X))
    return X_train,y_train

def sigmoid(X):
    return 1./(1+np.exp(-X))

def stochastic_gradient(Xn,yn,W):
    SGD = -yn*Xn*sigmoid(-yn*np.dot(Xn,W))
    return SGD.reshape((-1,1))

def batch_gradient(X,y,W):
    BGD = np.mean((-y*X)*sigmoid(-y*np.dot(X,W)),0)
    return BGD.reshape((-1,1))

def train(X,y,iteration=2000,eta=0.001,SGD=True):
    W = np.random.random((X.shape[1],1))
    index = 0
    for i in range(iteration):
        if index==X.shape[0]:
            index = 0
        if SGD:
            W = W - eta*stochastic_gradient(X[index],y[index],W)
        else:
            W = W - eta*batch_gradient(X,y,W)
        index +=1

    return W

def predict(X,y,W):
    y_pred = sigmoid(np.dot(X,W))
    y_pred = np.where(y_pred<0.5,-1,1)
    count = np.sum(y_pred!=y)
    return count/X.shape[0]



if __name__ == '__main__':
    train_set = 'E:/programming/python/ML_NTU/机器学习基石/hw3_train.dat'
    test_set = 'E:/programming/python/ML_NTU/机器学习基石/hw3_test.dat'
    X,y = data_preprocessing(train_set)
    W = train(X,y,eta=0.01,SGD= False)
    X_test,y_test = data_preprocessing(test_set)
    E_in = predict(X,y,W)
    print(E_in)
    E_out = predict(X_test,y_test,W)
    print(E_out)