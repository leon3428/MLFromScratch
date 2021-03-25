import numpy as np
import random

dataFile = 'data_banknote_authentication.txt' 

def loadData(file):
    x = []
    y = []

    with open(file) as f:
        data = f.read()
    data = data.split()
    random.shuffle(data)
    
    for line in data:
        line = list(map(float, line.split(',')))
        x.append(line[:4])
        y.append(line[4])


    x_train = x[:1000]
    y_train = y[:1000]
    x_test = x[1000:]
    y_test = y[1000:]
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test) 
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    
    return x_train, y_train, x_test, y_test 

class LogisticRegression:
    def __init__(self, input_dim = 1):
        self.w = np.ones((input_dim+1,1))

    def __sigmoid(self, v):
        return 1/(1+np.exp(-v))

    def fit(self, x, y, lr, epochs, verbose = True, ):
        X0 = np.ones((x.shape[0],1))
        X = np.hstack((X0, x))
        assert X.shape[1] == self.w.shape[0], "matrix x does not match input_dim"
        assert X.shape[0] == y.shape[0], "matrix x and vector y have different number of samples"

        for i in range(epochs):
            nabla = (1/y.shape[0]) * np.transpose(X) @ (y - self.__sigmoid(X @ self.w))
            
            self.w += lr * nabla 

            if verbose:
                print(f"[Epoch: {i+1}/{epochs}]: accuracy: {self.evaluate(x, y)*100}%") 
    
    def evaluate(self, X, y):
        X0 = np.ones((X.shape[0],1))
        X = np.hstack((X0, X))
        assert X.shape[1] == self.w.shape[0], "matrix x does not match input_dim"
        assert X.shape[0] == y.shape[0], "matrix x and vector y have different number of samples"

        predictions = self.__sigmoid(X @ self.w)

        diff = np.abs(predictions - y)
        n_correct = (diff < 0.5).sum()

        return n_correct / y.shape[0]
        

def main():
    x_train, y_train, x_test, y_test = loadData(dataFile)
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    lr = LogisticRegression(4)
    lr.fit(x_train, y_train, 1, 30)

    print(lr.evaluate(x_test, y_test))

if __name__ == '__main__':
    main()