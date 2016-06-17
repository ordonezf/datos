import numpy as np
import csv

LAPS = 60000

def sigmoid(x,derive=False):
    if derive:
        return x*(1-x)

    return 1/(1+np.exp(-x))
    

np.random.seed(0)
derive = True
train = open("csv/train.csv", "r")
r = csv.reader(train)
next(r)

data = []
target = []

for row in r:
    aux = [0 for x in xrange(10)]
    aux[int(row[0])] = 1
    target.append(aux)
    data.append([float(x) for x in row[1:]])

train.close()

data = np.array(data)
target = np.array(target)

w1 = 2 * np.random.random((len(data[0]), 50)) - 1   #[40k x 50]
w2 = 2 * np.random.random((50, 25)) - 1             #[50 x 25]
w3 = 2 * np.random.random((25, 10)) - 1             #[25 x 10]

for lap in xrange(LAPS):
    print lap
    layer_0 = data
    layer_1 = sigmoid(np.dot(layer_0, w1), not derive)
    layer_2 = sigmoid(np.dot(layer_1, w2), not derive)
    layer_3 = sigmoid(np.dot(layer_2, w3), not derive)

    layer_3_error = target - layer_3
    layer_3_delta = layer_3_error * sigmoid(layer_3, derive)

    layer_2_error = np.dot(layer_3_delta, w3.T)
    layer_2_delta = layer_2_error * sigmoid(layer_2, derive)

    layer_1_error = np.dot(layer_2_delta, w2.T)
    layer_1_delta = layer_1_error * sigmoid(layer_1, derive)

    w3 = np.dot(layer_2.T, layer_3_delta)
    w2 = np.dot(layer_1.T, layer_2_delta)
    w1 = np.dot(layer_0.T, layer_1_delta)

print "w1:", w1

print "w2:", w2

print "w3:", w3






