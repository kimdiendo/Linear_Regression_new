from csv import reader
import numpy as np
import pandas as pd
import math


# function is to load CSVfile
def load_data(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset


# function is to convert all row of each column into float numbers
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# load CSV file
filename = 'winequality-white.csv'
dataset = load_data(filename)
# convert _string_to_float
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# function is to find maximum value of each column
a = []  # Create NULL array
for i in range(len(dataset[0])):
    max = dataset[0][i]
    for j in range(len(dataset)):
        if max <= dataset[j][i]:
            max = dataset[j][i]
    a.append(max)
# function is to find minimum value of each column
b = []  # Create NULL array
for i in range(len(dataset[0])):
    min = dataset[0][i]
    for j in range(len(dataset)):
        if min >= dataset[j][i]:
            min = dataset[j][i]
    b.append(min)
# To calculate min,max normalization
for i in range(len(dataset[0])):
    for j in range(len(dataset)):
        dataset[j][i] = (dataset[j][i] - b[i]) / (a[i] - b[i])  # Using fomula of min-max normalization
datatrain = dataset[0: int(len(dataset) * 0.8)]
data_test = dataset[int(len(dataset) * 0.8): len(dataset)]
datatrain = pd.read_csv("test.csv")
datatrain = datatrain.values
datatrain = datatrain.T
X_train = datatrain[0:11]
Y_train = datatrain[11]


def model(X_train, Y_train, learning_rate):
    m = X_train.shape[1]
    n = X_train.shape[0]
    W = np.random.random(n)
    B = 0
    while True:
        Y = np.dot(W.T, X_train) + B
        Loss_function = (1 / m) * np.sum(np.multiply(Y - Y_train, Y - Y_train))
        print(Loss_function)
        if Loss_function < 0.02:
            break
        dW = (1 / m) * np.dot((2 * (Y - Y_train)), X_train.T)
        dB = (1 / m) * np.sum(2 * (Y - Y_train))
        W = W - learning_rate * dW.T
        B = B - learning_rate * dB
    return W, B


learning_rate = 0.001
list_loss, B = model(X_train, Y_train, learning_rate)
print("Configuration of Loss Function :")
print(list_loss)
print(B)
# Test 20% dataset
y = []
for i in range(len(data_test)):
    y1 = 0.0
    for j in range(len(list_loss)):
        y1 = y1 + list_loss[j] * data_test[i][j]
    y.append(math.pow((y1 + B - data_test[i][len(list_loss)]), 2))
loss = (sum(y)) / (len(data_test))
print("Value of Mean spared root " + str(loss))
for i in range(len(data_test)):
    print(str(data_test[i][len(list_loss)]) + " : " + str(y[i]))


