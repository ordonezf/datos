import csv

'''
Class that encapsulates a neuron, has weights, that are its internal
state, threshold centers the training of the dataset in a specific
position, this is a hyperparameter and depends exclusivly on the data.
LearningRate, is another hyperparameter
'''
class Perceptron(object):
    weights = []
    threshold = 0
    learningRate = 0

    WEIGHTS_LABEL = "weights"
    THRESHOLD_LABEL = "threshold"
    LEARNING_RATE_LABEL = "learningRate"

    @staticmethod
    def perceptronWithFilename(filename):
        perceptron = Perceptron()
        f = open(filename + ".csv", 'r')
        try:
            reader = csv.reader(f)
            for row in reader:
                if (row[0] == perceptron.WEIGHTS_LABEL):
                    perceptron.weights = map(lambda x : float(x), row[1:])
                if (row[0] == perceptron.THRESHOLD_LABEL):
                    perceptron.threshold = float(row[1])
                if (row[0] == perceptron.LEARNING_RATE_LABEL):
                     perceptron.learningRate = float(row[1])
        finally:
            f.close()

        print "weights :", perceptron.weights, "threshold :", perceptron.threshold, "learningRate :", perceptron.learningRate

        return perceptron

    def __init__(self, dimension = 100, threshold = 0.5, learningRate = 0.1):
        self.weights = [0 for x in xrange(dimension)]
        self.threshold = threshold
        self.learningRate = learningRate

    def dotProduct(self, values):
        return sum(valor * weight for valor, weight in zip(values, self.weights))

    def train(self, trainingSet, limitIterations):
        while True:
            errorSet = 0
            limitIterations -= 1

            for vector_de_entrada, salida_deseada in trainingSet:
                result = self.dotProduct(vector_de_entrada) > self.threshold
                error = salida_deseada - result
                if error != 0:
                    errorSet += 1
                    for indice, valor in enumerate(vector_de_entrada):
                        self.weights[indice] += self.learningRate * error * valor
            if (errorSet == 0) or (limitIterations == 0):
                break
        return

    def test(self, inputValue):
        return (self.dotProduct(inputValue) - self.threshold) > 0

    def saveWeights(self, filename):
        weightsCsv = self.weights[:]
        weightsCsv.insert(0, "weights")

        thresholdCsv = ["threshold", self.threshold]
        learningRateCsv = ["learningRate", self.learningRate]

        weightsFile = open(filename + ".csv", 'w')
        writer = csv.writer(weightsFile)
        writer.writerows([weightsCsv, thresholdCsv, learningRateCsv])
        weightsFile.close()

'''
Small demostration of how perceptron works
'''
def test():
    threshold = 0.5
    leaarningRate = 0.1
    pesos = [0, 0, 0]
    trainingSet = [((1, 0, 0), 1), ((1, 0, 1), 1), ((1, 1, 0), 1), ((1, 1, 1), 0)]


    limitIterations = 100000

    perceptron = Perceptron(3, threshold, leaarningRate)
    perceptron.train(trainingSet, limitIterations)

    print perceptron.test((0, 1, 0))
