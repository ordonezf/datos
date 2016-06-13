
class Perceptron(object):
    wights = []
    threshold = 0
    learningRate = 0;


    def __init__(self, dimension, threshold, learningRate):
        self.weights = [0 for x in xrange(dimension)]
        self.threshold = threshold
        self.learningRate = learningRate

    def dotProduct(self, values):
        return sum(valor * weight for valor, weight in zip(values, self.weights))

    def train(self, trainingSet, limitIterations):
        while True:
            print('-' * 60)
            errorSet = 0
            limitIterations -= 1

            for vector_de_entrada, salida_deseada in trainingSet:
                print(self.weights)
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


def test():
    threshold = 0.5
    leaarningRate = 0.1
    pesos = [0, 0, 0]
    trainingSet = [((1, 0, 0), 1), ((1, 0, 1), 1), ((1, 1, 0), 1), ((1, 1, 1), 0)]


    limitIterations = 100000

    perceptron = Perceptron(3, threshold, leaarningRate)
    perceptron.train(trainingSet, limitIterations)

    print perceptron.test((0, 1, 0))
