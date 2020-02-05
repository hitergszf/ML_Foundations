import numpy as np
def generate_data():
    X = np.zeros((1000,2))
    for i in range(1000):
        x = np.random.uniform(-1,1,2).reshape(1,-1)
        X[i,:] = x
    y = np.sign(np.sum(X ** 2, 1) - 0.6)
    X = np.insert(X, 0, 1, axis=1)
    noise = np.where(np.random.random(X.shape[0])<0.1,-1,1)
    noise_y = noise*y
    return X,noise_y

def linear_regression(X,y,lam=0):
    W = np.linalg.inv(np.dot(X.T,X)+lam*np.eye(X.shape)).dot(X.T).dot(y)
    return W

def calculate_E_in(X,y,W):
    error = np.where(np.dot(X,W)*y<0,1,0)
    return np.sum(error)*1./X.shape[0]

if __name__ == '__main__':
    T = 1000
    sum_E_in = 0
    for i in range(T):
        X,y = generate_data()
        W = linear_regression(X,y)
        E_in = calculate_E_in(X,y,W)
        sum_E_in += E_in

    print(sum_E_in/T)
    #0.5048459999999998
