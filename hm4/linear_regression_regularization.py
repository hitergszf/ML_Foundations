import numpy as np
import pandas as pd

def data_preprocessing(path):
    data = pd.read_table(path,sep = '\s',header=None, engine='python')
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    y_train = y.reshape((-1,1))
    X_train = np.hstack((np.ones((X.shape[0],1),dtype=int),X))
    return X_train,y_train

def linear_regression(X,y,lam=0):
    W = np.linalg.inv(np.dot(X.T,X)+lam*np.eye(X.shape)).dot(X.T).dot(y)
    return W

def predict(X,y,W):
    error = np.where(np.dot(X,W)*y<0,1,0)
    return np.sum(error)*1./X.shape[0]

def linear_regression(X,y,lam=0):
    W = np.linalg.inv(np.dot(X.T,X)+lam*np.eye(X.shape[1])).dot(X.T).dot(y)
    return W

if __name__ == '__main__':
    train_set = 'E:/programming/python/ML_NTU/机器学习基石/hw4_train.dat'
    test_set = 'E:/programming/python/ML_NTU/机器学习基石/hw4_test.dat'
    X,y = data_preprocessing(train_set)
    X_test, y_test = data_preprocessing(test_set)
    X_train,y_train = X[:120],y[:120]
    X_val,y_val = X[120:],y[120:]
    lams = np.arange(-10,3)
    lams = 10.**lams
    best_lam = -100.
    best_E_train = 100.

    best_E_val = 100.

    for lam in lams:
        W = linear_regression(X_train,y_train,lam=lam)
        E_train = predict(X_train, y_train, W)
        E_val = predict(X_val,y_val,W)

        if E_val <= best_E_val: #相等的时候更新为更大的那个
            best_E_train = E_train
            best_E_val = E_val
            best_lam = lam
    W = linear_regression(X,y,lam = best_lam)
    E_in = predict(X,y,W)
    E_out = predict(X_test,y_test,W)
    print(E_in,E_out)