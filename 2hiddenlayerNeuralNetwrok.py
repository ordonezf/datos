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
    def __init__(self, input, hidden1, hidden2, output, iterations, learning_rate, momentum, rate_decay):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate
        self.momentum = momentum
        self.rate_decay = rate_decay

        # initialize arrays
        self.input = input
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output = output

        # set up array of 1s for activations
        self.ai = 1
        self.ah1 = 1
        self.ah2 = 1
        self.ao = 1

        # create randomized weights
        # use scheme from 'efficient backprop to initialize weights
        input_range = 1.0 / self.input ** (1/2)
        hidden_range = 1.0 / self.hidden1 ** (1/2)
        self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden1))
        self.wh = np.random.normal(loc = 0, scale = hidden_range, size = (self.hidden1, self.hidden2))
        self.wo = np.random.uniform(size = (self.hidden2, self.output)) / np.sqrt(self.hidden2)
        

        # create arrays of 0 for changes
        # this is essentially an array of temporary values that gets updated at each iteration
        # based on how much the weights need to change in the following iteration
        self.ci = np.zeros((self.input, self.hidden1))
        self.ch = np.zeros((self.hidden1, self.hidden2))
        self.co = np.zeros((self.hidden2, self.output))

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
        self.ah1 = tanh(self.ai.dot(self.wi))
        self.ah2 = tanh(self.ah1.dot(self.wh))
        self.ao = softmax(self.ah2.dot(self.wo))

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
        hidden2_deltas = dtanh(self.ah2) * error

        error = hidden2_deltas.dot(self.wh.T)
        hidden1_deltas = dtanh(self.ah1) * error

        ############output ----> hidden_2##############
        change = output_deltas.T.dot(self.ah2).T
        self.wo -= (self.learning_rate * change) + (self.co * self.momentum)
        self.co = change
        ############hidden_2 ----> hidden_1##############
        change = hidden2_deltas.T.dot(self.ah1).T
        self.wh -= (self.learning_rate * change) + (self.ch * self.momentum)
        self.ch = change
        ############hidden_1 ----> input##############
        change = hidden1_deltas.T.dot(self.ai).T
        self.wi -= (self.learning_rate * change) + (self.ci * self.momentum)
        self.ci = change

        return np.mean(-output_deltas)

    def train(self, patterns):
        print "Begin training"
        for i in range(self.iterations):
            error = 0.0
            self.feedForward(patterns[1])
            error = self.backPropagate(patterns[0])
            print "Error : {}, lap :  {}".format(error, i)
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))


    def test_cross(self, test):

        print "Predicting..."
        self.ai = np.array(test[1])
        self.ah1 = tanh(self.ai.dot(self.wi))
        self.ah2 = tanh(self.ah1.dot(self.wh))
        self.ao = softmax(self.ah2.dot(self.wo))

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
        self.ah1 = tanh(self.ai.dot(self.wi))
        self.ah2 = tanh(self.ah1.dot(self.wh))
        self.ao = softmax(self.ah2.dot(self.wo))

        e = 0
        for out, csv_out in zip(self.ao, ta):
            n = out.tolist().index(max(out))
            if n == int(csv_out[1]):
                e += 1

        print "{} laps  lr = {} momentum = {}  decay = {} Aciertos = {}".format(self.iterations, self.learning_rate_init, self.momentum, self.rate_decay, e/28000.0)
        print e
        test.close()
        ar.close()

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
        self.ah1 = tanh(self.ai.dot(self.wi))
        self.ah2 = tanh(self.ah1.dot(self.wh))
        self.ao = softmax(self.ah2.dot(self.wo))

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

        target = np.array(target)

        #train = [target[:35000],data[:35000]]
        #test = [target[35000:],data[35000:]]

        return [target, data]

    NN = MLP_NeuralNetwork(784, 250, 100, 10,
        iterations = 50,
        learning_rate = 0.2,
        momentum = 0.05,
        rate_decay = 0.005)

    train = load_data()

    NN.train(train)
    #NN.test_cross(test)
    #NN.test()
    NN.test_against()

if __name__ == '__main__':
    demo()

#cross-validation
    # 15 laps -> lr -> 0.1  ->  Aciertos: 0.072058823529
    # 15 laps -> lr -> 0.5  ->  Aciertos: 0.052117647058
    # 15 laps -> lr -> 0.01 ->  Aciertos: 0.046
    # 50 laps -> lr -> 0.01 ->  Aciertos: 0.182529411765
    # 50 laps -> lr -> 0.1  ->  Aciertos: 0.300823529412
    # 50 laps -> lr -> 0.1  ->  Aciertos: 0.325764705882    -> l2_in = 0.01 -> l2_out = 0.01
    # 50 laps -> lr -> 0.1  ->  Aciertos: 0.328117647059    -> l2_in = 0.1  -> l2_out = 0.1
    # 50 laps -> lr -> 0.1  ->  Aciertos: 0.042117647058    -> l2_in = 10   -> l2_out = 10
    # 50 laps -> lr -> 0.1  ->  Aciertos: 0.225352941176    -> l2_in = 0.5  -> l2_out = 0.5
    # 50 laps -> lr -> 0.1  ->  Aciertos: 0.220764705882    -> l2_in = 0.5  -> l2_out = 0.1
    # 50 laps -> lr -> 0.1  ->  Aciertos: 0.297705882353    -> l2_in = 0.1  -> l2_out = 0.5
    # 50 laps -> lr -> 0.1  ->  Aciertos: 0.267235294118    -> l2_in = 0.1  -> l2_out = 0.1     -> hl = 200
    # 50 laps -> lr -> 0.1  ->  Aciertos: 0.336823529412    -> l2_in = 0.1  -> l2_out = 0.1     -> hl = 150
    # 50 laps -> lr -> 0.1  ->  Aciertos: 0.350058823529    -> l2_in = 0.1  -> l2_out = 0.1     -> momentum = 0.01
    # 50 laps -> lr -> 0.05 ->  Aciertos: 0.349882352941    -> l2_in = 0.1  -> l2_out = 0.1     -> momentum = 0.005  -> decay = 0.001
    # 50 laps -> lr -> 0.05 ->  Aciertos: 0.338588235294    -> l2_in = 0.1  -> l2_out = 0.1     -> momentum = 0.05   -> decay = 0.001
    # 50 laps -> lr -> 0.05 ->  Aciertos: 0.343823529412    -> l2_in = 0.1  -> l2_out = 0.1     -> momentum = 0.05   -> decay = 0.01
    # 50 laps -> lr -> 0.05 ->  Aciertos: 0.345882352941    -> l2_in = 0.1  -> l2_out = 0.1     -> momentum = 0.005   -> decay = 0.01

#1 hidden layer
    # 50 laps   lr = 0.5  momentum = 0.01 decay = 0.001  Aciertos: 0.435428571429
    # 100 laps  lr = 0.5  momentum = 0.01 decay = 0.001  Aciertos: 0.601928571429
    # 100 laps  lr = 0.01 momentum = 0.5  decay = 0.0001 Aciertos: 0.8425
    # 100 laps  lr = 0.05 momentum = 0.5  decay = 0.0001 Aciertos: 0.823642857143
    # 100 laps  lr = 0.05 momentum = 0.1  decay = 0.0001 Aciertos: 0.737785714286
    # 100 laps  lr = 0.01 momentum = 0.5  decay = 0.0002 Aciertos = 0.82725
    # 100 laps  lr = 0.01 momentum = 0.8  decay = 0.0002 Aciertos = 0.844607142857
    # 100 laps  lr = 0.01 momentum = 1    decay = 0.0002 Aciertos = 0.8265
    # 100 laps  lr = 0.01 momentum = 1.5  decay = 0.0002 Aciertos = 0.827571428571
    # 100 laps  lr = 0.01 momentum = 0.8  decay = 0.0005 Aciertos = 0.857142857143
    # 150 laps  lr = 0.01 momentum = 0.8  decay = 0.0002 Aciertos = 0.771107142857
    # 100 laps  lr = 0.01 momentum = 0.8  decay = 0.0005 Aciertos = 0.823464285714
        #layers: 784 1000 10

#2 hidden layer
    # 50 laps  lr = 0.3 momentum = 0.05  decay = 0.01 Aciertos = 0.665  784, 250, 100, 10
    # 50 laps  lr = 0.5 momentum = 0.05  decay = 0.001 Aciertos = 0.651285714286   784, 250, 100, 10
    # 50 laps  lr = 0.01 momentum = 0.8  decay = 0.0005 Aciertos = 0.114178571429   784, 50, 200, 10
    # 50 laps  lr = 0.01 momentum = 0.8  decay = 0.0005 Aciertos = 0.128892857143    784, 225, 95, 10
    # 50 laps  lr = 0.5 momentum = 0.8  decay = 0.0005 Aciertos = 0.416964285714    784, 225, 95, 10
    # 50 laps  lr = 0.5 momentum = 0.5  decay = 0.05 Aciertos = 0.505321428571    784, 250, 100, 10
