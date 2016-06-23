import math
import random
import numpy as np
import csv

np.seterr(all = 'ignore')
np.random.seed(0)

# sigmoid transfer function
# IMPORTANT: when using the logit (sigmoid) transfer function for the output layer make sure y values are scaled from 0 to 1
# if you use the tanh for the output then you should scale between -1 and 1
# we will use sigmoid for the output layer and tanh for the hidden layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def dsigmoid(y):
    return y * (1.0 - y)

# using tanh over logistic sigmoid is recommended
def tanh(x):
    return np.tanh(x)

# derivative for tanh sigmoid
def dtanh(x):
    y = tanh(x)
    return 1 - y*y

def softmax(x):
    e = [np.exp(ex - np.amax(ex)) for ex in x]
    out = [e1 / np.sum(e1) for e1 in e]
    return np.array(out)

class MLP_NeuralNetwork(object):
    """
    Basic MultiLayer Perceptron (MLP) network, adapted and from the book 'Programming Collective Intelligence' (http://shop.oreilly.com/product/9780596529321.do)
    Consists of three layers: input, hidden and output. The sizes of input and output must match data
    the size of hidden is user defined when initializing the network.
    The algorithm has been generalized to be used on any dataset.
    As long as the data is in this format: [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           ...
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]]]
    An example is provided below with the digit recognition dataset provided by sklearn
    Fully pypy compatible.
    """
    def __init__(self, input, hidden, output, iterations, learning_rate, momentum, rate_decay):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rate_decay = rate_decay

        # initialize arrays
        self.input = input # add 1 for bias node
        self.hidden = hidden
        self.output = output

        # set up array of 1s for activations
        self.ai = 1#np.random.random((42000, self.input))
        self.ah = 1#np.random.random((self.input, self.hidden))
        self.ao = 1#np.random.random((self.hidden, self.output))

        # create randomized weights
        # use scheme from 'efficient backprop to initialize weights
        input_range = 1.0 / self.input ** (1/2)
        output_range = 1.0 / self.hidden ** (1/2)
        self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden))
        self.wo = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)
        

        # create arrays of 0 for changes
        # this is essentially an array of temporary values that gets updated at each iteration
        # based on how much the weights need to change in the following iteration
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        """
        The feedForward algorithm loops over all the nodes in the hidden layer and
        adds together all the outputs from the input layer * their weights
        the output of each node is the sigmoid function of the sum of all inputs
        which is then passed on to the next layer.
        :param inputs: input data
        :return: updated activation output vector
        """
        self.ai = np.array(inputs)
        self.ah = tanh(self.ai.dot(self.wi))
        self.ao = softmax(self.ah.dot(self.wo))

    def backPropagate(self, targets):
        """
        For the output layer
        1. Calculates the difference between output value and target value
        2. Get the derivative (slope) of the sigmoid function in order to determine how much the weights need to change
        3. update the weights for every node based on the learning rate and sig derivative
        For the hidden layer
        1. calculate the sum of the strength of each output link multiplied by how much the target node has to change
        2. get derivative to determine how much weights need to change
        3. change the weights based on learning rate and derivative
        :param targets: y values
        :param N: learning rate
        :return: updated weights
        """
        target = np.array(targets)
        output_deltas = -(target - self.ao)

        error = output_deltas.dot(self.wo.T)
        hidden_deltas = dtanh(self.ah) * error

        change = output_deltas.T.dot(self.ah).T
        #l2_out = 0.01 * self.wo
        self.wo -= self.learning_rate * (change) + self.co * self.momentum
        self.co = change

        change = hidden_deltas.T.dot(self.ai).T
        #l1_out = 0.01 * self.wi
        self.wi -= self.learning_rate * (change) + self.ci * self.momentum
        self.ci = change

        return np.mean(-output_deltas)

    def train(self, patterns):
        # N: learning rate
        print "Begin training"
        for i in range(self.iterations):
            error = 0.0
            self.feedForward(patterns[1])
            error = self.backPropagate(patterns[0])
            #with open('error.txt', 'a') as errorfile:
            #    errorfile.write(str(error) + '\n')
            #    errorfile.close()
            print "Error : {}, lap :  {}".format(error, i)

            # learning rate decay
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))


    def test_cross(self, test):

        self.ai = np.array(test[1])
        self.ah = tanh(self.ai.dot(self.wi))
        self.ao = softmax(self.ah.dot(self.wo))

        dic = {}
        c = 0
        for out,check in zip(self.ao,test[0]):
            n = out.tolist().index(max(out))
            if n == check.tolist().index(max(check)):
                c += 1

        print "Aciertos:", c/17000.0
        print c

    def test(self):
        """
        Currently this will print out the targets next to the predictions.
        Not useful for actual ML, just for visual inspection.
        """
        test = open("csv/test.csv", "r")
        r = csv.reader(test)
        next(r)
        ar = open("csv/submit2.csv","w")
        w = csv.writer(ar)

        print self.wi[0].mean()
        print self.wo[0].mean()
        print "Predicting..."
        output = []
        self.ai = []
        for row in r:
            self.ai.append([int(x) for x in row])
        self.ai = np.array(self.ai)
        self.ah = tanh(self.ai.dot(self.wi))
        self.ao = softmax(self.ah.dot(self.wo))

        w.writerow(("ImageId","Label"))
        c = 1
        e = 0
        dic = {}
        for out in self.ao:
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
        test.close()
        ar.close()

    def predict(self, X):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))
        return predictions

def demo():
    """
    run NN demo on the digit recognition dataset from sklearn
    """
    def load_data():
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
        data -= data.min() # scale the data so values are between 0 and 1
        data /= data.max() # scale

        target = np.array(target)

        train = [target[:35000],data[:35000]]
        test = [target[35000:],data[35000:]]

        return train, test

    NN = MLP_NeuralNetwork(784, 100, 10, iterations = 50, learning_rate = 0.01, momentum = 0.5, rate_decay = 0.01)

    train, test = load_data()

    NN.train(train)
    NN.test_cross(test)
    #NN.test(test)

if __name__ == '__main__':
    demo()


# 15 laps lr    -> 0.1  ->  Aciertos: 0.0720588235294
# 15 laps lr    -> 0.5  ->  Aciertos: 0.0521176470588
# 15 laps lr    -> 0.01 ->  Aciertos: 0.046
# 50 laps lr    -> 0.01 ->  Aciertos: 0.182529411765