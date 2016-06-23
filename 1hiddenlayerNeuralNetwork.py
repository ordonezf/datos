import math
import random
import numpy as np
import csv

np.seterr(all = 'ignore')
#np.random.seed(0)

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

class NeuralNetwork(object):
    """
    3 layer neural network
    """
    def __init__(self, input, hidden, output, iterations, learning_rate, momentum, rate_decay):
        """
        input: number of input neurons
        hidden: number of hidden neurons
        output: number of output neurons
        """
        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate
        self.momentum = momentum
        self.rate_decay = rate_decay
        self.input = input
        self.hidden = hidden
        self.output = output

        self.ai = np.zeros(self.input)
        self.ah = np.zeros(self.hidden)
        self.ao = np.zeros(self.output)

        # create randomized weights
        # use scheme from 'efficient backprop to initialize weights
        input_range = 1.0 / self.input ** (1/2)
        self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden))
        self.wo = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)
        

        #array of past iteration weights that change the current weight with the momentum
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        """
        inputs: input data
        """
        self.ai = np.array(inputs)
        self.ah = tanh(self.ai.dot(self.wi))
        self.ao = softmax(self.ah.dot(self.wo))

    def backPropagate(self, targets):
        """
        :param targets: y values
        :return: error
        """
        target = np.array(targets)
        output_deltas = -(target - self.ao)

        #output_error ---backPropagate--->  hidden_layer
        error = output_deltas.dot(self.wo.T)
        hidden_deltas = dtanh(self.ah) * error

        #update weights
        change = output_deltas.T.dot(self.ah).T
        self.wo -= (self.learning_rate * change) + (self.co * self.momentum)
        self.co = change

        change = hidden_deltas.T.dot(self.ai).T
        self.wi -= (self.learning_rate * change) + (self.ci * self.momentum)
        self.ci = change

        return np.mean(-output_deltas)

    def train(self, patterns):
        print "Begin training"
        for i in range(self.iterations):
            self.feedForward(patterns[1])
            error = self.backPropagate(patterns[0])
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))
            print "Error: {}, lap: {}".format(error, i)


    def test_cross(self, test):

        self.ai = np.array(test[1])
        self.ah = tanh(self.ai.dot(self.wi))
        self.ao = softmax(self.ah.dot(self.wo))

        dic = {}
        c = 0
        e = 0
        for out,check in zip(self.ao,test[0]):
            e += 1
            n = out.tolist().index(max(out))
            if n == check.tolist().index(max(check)):
                c += 1

        print "Aciertos:", c/float(e)

    def test_against(self):
        test = open("csv/test.csv", "r")
        r = csv.reader(test)
        next(r)
        ar = open("csv/submit98.csv","r")
        ta = csv.reader(ar)
        next(ta)

        print "Predicting..."
        output = []
        self.ai = []
        for row in r:
            self.ai.append([float(x) for x in row])
        self.ai = np.array(self.ai)
        self.ah = tanh(self.ai.dot(self.wi))
        self.ao = softmax(self.ah.dot(self.wo))

        e = 0
        c = 0
        for out, csv_out in zip(self.ao, ta):
            c += 1
            n = out.tolist().index(max(out))
            if n == int(csv_out[1]):
                e += 1

        print "{} laps  lr = {} momentum = {}  decay = {} Aciertos = {}".format(self.iterations, self.learning_rate_init, self.momentum, self.rate_decay, e/float(c))
        print e
        test.close()
        ar.close()

    def test(self):
        test = open("csv/test.csv", "r")
        r = csv.reader(test)
        next(r)
        ar = open("csv/submit.csv","w")
        w = csv.writer(ar)

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


def run():
    """
    run the NN on the mnist data set
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
            data.append([int(x) for x in row[1:]])

        train.close()

        data = np.array(data)

        target = np.array(target)

        #uncomment for cross-validation
        #train = [target[:35000],data[:35000]]
        #test = [target[35000:],data[35000:]]
        #return train, test

        return [target, data]

    NN = NeuralNetwork(784, 100, 10,
        iterations = 100,
        learning_rate = 0.001,
        momentum = 0.8,
        rate_decay = 0.0005)

    train = load_data()
    NN.train(train)
    NN.test_against()
    #NN.test()

    #uncomment for cross-validations
    #train,test = load_data()
    #NN.test_cross(test)

if __name__ == '__main__':
    run()


    # 100 laps  lr = 0.1 momentum = 0.5  decay = 0.01 Aciertos = 0.853678571429    100
    # 100 laps  lr = 0.5 momentum = 0.05  decay = 0.001 Aciertos = 0.834785714286  100
    # 100 laps  lr = 0.01 momentum = 0.8  decay = 0.0005 Aciertos = 0.873178571429 100
    # 100 laps  lr = 0.01 momentum = 0.8  decay = 0.0005 Aciertos = 0.871821428571  150
    # 100 laps  lr = 0.5 momentum = 0.8  decay = 0.0005 Aciertos = 0.849785714286   100
