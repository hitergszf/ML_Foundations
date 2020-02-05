import numpy as np

def generate_data(time_seed):
    np.random.seed(time_seed)
    X = np.sort(np.random.uniform(-1, 1, 20))
    y = np.sign(X)
    noise = np.random.random(X.shape[0])
    noise_y = y*np.where(noise < 0.2, -1, 1)
    return X, noise_y


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

def E_out(sign,theta):
    return 0.5 + 0.3 * sign * (abs(theta) - 1)

if __name__ == '__main__':
    T = 5
    sum_E_in = 0
    sum_E_out = 0
    for t in range(T):
        X,y = generate_data(t)
        E_in,sign,theta = calculate_E_in(X,y)
        #print(E_in,sign,theta)
        sum_E_in += E_in
        sum_E_out += E_out(sign,theta)
    print(1.*sum_E_in/(20*T))
    print(1.*sum_E_out/T)
