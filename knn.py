import numpy as np
import pandas as pd

trainFile = 'mnist_train.csv'
testFile = 'mnist_test.csv'
K = 1

def loadDataset(fileName):
    print("Opening:", fileName)
    df = pd.read_csv(fileName)
    images = df.loc[:, df.columns != 'label']
    labels = df['label']
    labels = labels.to_numpy()
    images = images.to_numpy()

    print("X:", images.shape)
    print("Y:", labels.shape)

    return images,labels

def mostOccuring(l):
    cnt = [0]*10
    for i in l:
        cnt[i]+=1
    for i in range(10):
        if cnt[i] == max(cnt):
            return i

def predict(image, datasetX, datasetY):
    image = image.reshape((1, image.shape[0]))
    dif = datasetX - image
    dist = np.linalg.norm(dif, axis = 1)

    ind = dist.argsort()
    labels = datasetY[ind[:K]]

    return mostOccuring(labels)


def evaluate():
    x_train, y_train = loadDataset(trainFile)
    x_test, y_test = loadDataset(testFile)

    n_correct = 0
    for i in range(y_test.shape[0]):
        predicted_label = predict(x_test[i], x_train, y_train) 
        if predicted_label == y_test[i]:
            n_correct += 1
        print(f"{i}/{y_test.shape[0]}: Acc: {n_correct/(i+1)*100}%")

if __name__ == '__main__':
    evaluate()