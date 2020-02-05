import numpy as np
import pandas as pd
def data_preprocessing(path):
    data = pd.read_table(path,sep = '\s',header=None, engine='python')
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values 
    y_train = y.reshape((-1,1))
    return X,y


def hypothesis(theta, sign, X):
    return sign*np.sign(X-theta)


def calculate_E_in(X, y):
    thetas = [(-1+X[0])/2]
    for i in range(X.shape[0]-1):
        thetas.append((X[i]+X[i+1])/2)
    thetas.append((1+X[-1])/2)
    best_s = 0
    best_theta = -1
    E_in = X.shape[0]
    
    for theta in thetas:
        y_positive_predict = hypothesis(theta, 1, X)
        y_negative_predict = hypothesis(theta, -1, X)
        count_positive = np.sum(y != y_positive_predict)
        count_negative = np.sum(y != y_negative_predict)
        if count_negative < count_positive:
            if count_negative < E_in:
                E_in = count_negative
                best_s = -1
                best_theta = theta
        else:
            if count_positive < E_in:
                E_in = count_positive
                best_s = 1
                best_theta = theta
    
    return E_in,best_s,best_theta


if __name__ == "__main__":
    train_set = 'E:/programming/python/ML_NTU/机器学习基石/hw2_train.dat'
    test_set ='E:/programming/python/ML_NTU/机器学习基石/hw2_test.dat'
    X_train,y_train = data_preprocessing(train_set)
    E_in_best = X_train.shape[0]
    best_s = 1
    best_theta = 0
    index = -1
    for i in range(X_train.shape[1]):
        X = X_train[:,i]
        input_data = np.array([X,y_train])
        input_data = input_data[np.argsort(input_data[:,0])]
        E_in, sign, theta = calculate_E_in(X,y_train)
        if E_in < E_in_best:
            E_in_best = E_in
            best_s = sign
            best_theta = theta
            index = i
    print(E_in_best/X_train.shape[0])

    X_test,y_test = data_preprocessing(test_set)
    X_valid = X_test[:,index]
    predict_y = hypothesis(best_theta,best_s,X_valid)
    E_out = np.sum(predict_y!=y_test)
    print(E_out/X_valid.shape[0])
