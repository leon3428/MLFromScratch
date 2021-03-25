import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

train_file = 'boston_data.csv'
test_file = 'boston_test_data.csv'

#LSTAT, RM, PTRATIO

def load_data(filename):
    df = pd.read_csv(filename)

    features = ['lstat', 'rm']
    target = 'medv'

    df = df.apply(lambda a: 1/a if a.name == 'lstat' else a)
    x = df.loc[:, features].to_numpy()
    y = df[target].to_numpy()
    y = np.expand_dims(y, axis = 1)
    print(x.shape, y.shape)

    return x,y

ALPHA = 0.2

def evaluate(x, y, w):
    v = y - x @ w
    ret = v.T @ v
    return ret[0][0] / x.shape[0]

def fun(x,y, w):
    return x*w[0][0] + y*w[1][0]

def main():
    x, y = load_data(train_file)

    x_val = x[300:]
    y_val = y[300:]
    x = x[:300]
    y = y[:300]

    x_mean = x.mean()
    y_mean = y.mean()

    x -= x_mean
    y -= y_mean

    x_std = x.std()
    y_std = y.std()

    x /= x_std
    y /= y_std

    w = np.linalg.pinv(x.T@x + ALPHA*np.identity(2)) @ x.T @ y

    x_val -= x_mean
    y_val -= y_mean
    x_val /= x_std
    y_val /= y_std

    print(evaluate(x_val, y_val, w))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_val[:,0], x_val[:,1], y_val, color = 'red')

    xm = np.arange(-1.5, -0.5, 0.1)
    ym = np.arange(0.0, 2.0, 0.1)
    Xm, Ym = np.meshgrid(xm, ym)

    Zm = fun(Xm,Ym, w)

    ax.plot_wireframe(Xm, Ym, Zm)

    plt.show()
    

if __name__ == '__main__':
    main()