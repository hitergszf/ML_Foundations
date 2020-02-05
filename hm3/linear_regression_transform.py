import numpy as np


def generate_data():
    X = np.zeros((1000, 2))
    for i in range(1000):
        x = np.random.uniform(-1, 1, 2).reshape(1, -1)
        X[i, :] = x
    y = np.sign(np.sum(X ** 2, 1) - 0.6)
    X = np.insert(X, 0, 1, axis=1)
    noise = np.where(np.random.random(X.shape[0]) < 0.1, -1, 1)
    noise_y = noise * y
    return X, noise_y


def feature_transform(X):
    Z = np.zeros((X.shape[0], 6))
    Z[:, :3] = X[:, :3]
    Z[:, 3] = X[:, 1] * X[:, 2]
    Z[:, 4] = X[:, 1] ** 2
    Z[:, 5] = X[:, 2] ** 2
    return Z

def linear_regression(X, y):
    W = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
    return W


def calculate_E_in(X, y, W):
    error = np.where(np.dot(X, W) * y < 0, 1, 0)
    return np.sum(error) * 1. / X.shape[0]


if __name__ == '__main__':
    T = 1000
    min_E_in = 1.
    best_W = np.zeros((1000,6))

    for i in range(T):
        X, y = generate_data()
        X = feature_transform(X)
        W = linear_regression(X, y)
        E_in = calculate_E_in(X, y, W)
        if E_in < min_E_in:
            best_W = W
            min_E_in = E_in

    print(min_E_in)
    # 0.088
    print(best_W.astype(float))
    #[-1.07153841 -0.03702895  0.02002864  0.03529843  1.80013554  1.69298961]
    E_out = 0.
    for i in range(T):
        X_test,y_test = generate_data()
        Z_test = feature_transform(X_test)
        E_out += calculate_E_in(Z_test,y_test,best_W)
    print(E_out/T)
    #0.11710999999999977