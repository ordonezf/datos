import perceptron as pc
import random as rd

def getRandomZero(dimension, elementMax):
    negative = [0 for x in xrange(dimension / 2)]

    for x in xrange(dimension / 2):
        negative.append(rd.randrange(elementMax))

    return negative


def getRandopmPositive(dimension, elementMax):
    positive = [rd.randrange(elementMax) for x in xrange(dimension/2)]
    for x in xrange(dimension/2):
        positive.append(0)

    return positive

def createRandomDataSet(dimension, ammount, elementMax):
    trueSet = []
    falseSet = []

    for i in xrange(ammount / 2):
        positive = getRandopmPositive(dimension, elementMax)
        trueSet.append(tuple(positive))

    for i in xrange(ammount / 2):
        negative = getRandomZero(dimension, elementMax)
        falseSet.append(tuple(negative))

    trueSet = map(lambda x: (x, 1), trueSet)
    falseSet = map(lambda x: (x, 0), falseSet)

    trueSet.extend(falseSet)

    return trueSet

def testRandomDataSet(dimension, ammount, elementMax, perceptron):
    randomTest = createRandomDataSet(dimension, ammount, elementMax)
    correctResultCount = 0

    for test in randomTest:
        if perceptron.test(test[0]) == test[1]:
            correctResultCount += 1

    return correctResultCount


def main():
    #example of perceptron creation with meta data
    dimension = 100
    threshold = 0.3
    learningRate = 0.1
    perceptron = pc.Perceptron(dimension, threshold, learningRate)

    #training with sets as [((1, 2, 3, 0, 0, 0), 1), ((0, 0, 0, 3, 3, 8), 0)]
    ammount = 1000
    elementMax = 256
    randomTest = createRandomDataSet(dimension, ammount, elementMax)

    rd.shuffle(randomTest)

    perceptron.train(randomTest, dimension)

    testAmmount = 1000

    correctnes = testRandomDataSet(dimension, testAmmount, elementMax, perceptron)

    print "correctness :", correctnes, "over :", testAmmount

    perceptron.saveWeights("test")
    perceptron = pc.Perceptron.perceptronWithFilename("test")

main()
