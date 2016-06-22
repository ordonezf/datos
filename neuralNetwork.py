import numpy as np
import csv

LAPS = 60000

###################Activation Functions#############
def sigmoid(x,derive=False):
    if derive:
        return x * (1 - x)

    x_prime = np.array([(xi - min(xi))/(max(xi) - min(xi)) for xi in x])
    return 1 / (1 + np.exp(-x_prime))

def sigv2(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.amax(x))
    dist = e / np.sum(e)
    return dist

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    y = tanh(x)
    return 1 - y*y

####################################
np.random.seed(96478)
derive = True
train = open("csv/train.csv", "r")
r = csv.reader(train)
next(r)

data = []
target = []

print "Prepping data..."
for row in r:
    aux = [0 for x in xrange(10)]
    aux[int(row[0])] = 1
    target.append(aux)
    data.append([float(x) for x in row[1:]])

train.close()

data = np.array(data)
target = np.array(target)

w1 =  np.random.normal(loc = 0, size = (len(data[0]), 50))   #[784 x 50]
w2 =  np.random.normal(loc = 0, size = (50, 10))             #[55 x 10]

learning_rate = 0.01
decay = 0.0001
c1 = np.zeros((len(data[0]), 50))
c2 = np.zeros((50,10))
layer_0 = data
print layer_0.shape
print "Training..."
for lap in xrange(10):
    ######################ForwardPropagation##################
    layer_1 = tanh(layer_0.dot(w1))
    layer_2 = softmax(layer_1.dot(w2))

    ######################BackPropagation#####################
    layer_2_error = -(target - layer_2)
    layer_2_delta = layer_2_error

    layer_1_error = layer_2_delta.dot(w2.T)
    layer_1_delta = dtanh(layer_1) * layer_1_error

    change = layer_2_delta.T.dot(layer_1).T
    w2 = (learning_rate * change) + c2
    c2 = change

    change = layer_1_delta.T.dot(layer_0).T
    w1 = (learning_rate * change) + c1
    c1 = change

    print "Lap {} error: {}".format(lap, -layer_2_error.mean())
    learning_rate = learning_rate * (learning_rate / (learning_rate + (learning_rate * decay)))

print w1.mean()
print "-----------------------------------------------------"
print w2.mean()
test = open("csv/test.csv", "r")
r = csv.reader(test)
next(r)
ar = open("csv/submit.csv","w")
w = csv.writer(ar)

print "Predicting..."
output = []
for row in r:
    layer_0 = np.array([int(x) for x in row])
    layer_1 = tanh(layer_0.dot(w1))
    layer_2 = softmax(layer_1.dot(w2))
    output.append(layer_2)

w.writerow(("ImageId","Label"))
c = 1
e = 0
dic = {}
for out in output:
    try:
        n = out.tolist().index(max(out))
        dic.setdefault(n,0)
        dic[n] += 1
        w.writerow((c, n))
    except:
        w.writerow((c, np.random.randint(0,9)))
        e += 1
    c += 1

print "Total errors: ",e
print dic
ar.close()
test.close()
