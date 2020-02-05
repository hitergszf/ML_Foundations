import numpy as np
import pandas as pd


def data_preprocessing(path):
    data = pd.read_table(path, sep='\s', header=None, engine='python')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y_train = y.reshape((-1, 1))
    X_train = np.hstack((np.ones((X.shape[0], 1), dtype=int), X))
    return X_train, y_train


def linear_regression(X, y, lam=0):
    W = np.linalg.inv(np.dot(X.T, X) + lam * np.eye(X.shape)).dot(X.T).dot(y)
    return W


def predict(X, y, W):
    error = np.where(np.dot(X, W) * y < 0, 1, 0)
    return np.sum(error) * 1. / X.shape[0]


def linear_regression(X, y, lam=0):
    W = np.linalg.inv(np.dot(X.T, X) + lam * np.eye(X.shape[1])).dot(X.T).dot(y)
    return W


if __name__ == '__main__':
    train_set = 'E:/programming/python/ML_NTU/机器学习基石/hw4_train.dat'
    test_set = 'E:/programming/python/ML_NTU/机器学习基石/hw4_test.dat'
    X, y = data_preprocessing(train_set)
    X_test, y_test = data_preprocessing(test_set)
    lams = np.arange(-10, 3)
    lams = 10. ** lams

    num_folds = 5
    num_data = int(X.shape[0] / num_folds)
    best_E_cv = 100.
    best_W = None
    for lam in lams:
        E_cv = 0.
        for i in range(num_folds):
            X_val = X[i * num_data:(i + 1) * num_data]
            y_val = y[i * num_data:(i + 1) * num_data]
            X_train = np.concatenate((X[:i * num_data], X[(i + 1) * num_data:]))
            y_train = np.concatenate((y[:i * num_data], y[(i + 1) * num_data:]))
            W = linear_regression(X_train, y_train,lam = lam)
            E_cv += predict(X_val, y_val, W)
        E_cv /= num_folds
        if E_cv <= best_E_cv:
            best_E_cv = E_cv
            best_lam = lam
            best_W = W

    print(best_lam, best_E_cv)
    E_in = predict(X,y,best_W)
    E_out = predict(X_test,y_test,best_W)
