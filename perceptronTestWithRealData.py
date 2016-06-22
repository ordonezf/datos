import perceptron as pc
import csv

'''
requestedNumber : is an integer and maps the training csv to something like
[((data), true), ...] and it is true when requestedNumber is actually the number
correspondinf to the data, in all other cases is false
'''
def createRealDataSet(requestedNumber):
    f = open("csv/train.csv", 'r')
    trainDataSet = []

    try:
        reader = csv.reader(f)
        line = 0
        for row in reader:
            if line == 0:
                line +=1
                continue
            # print "row", row
            isRequestedNumber = int(row[0]) == requestedNumber
            row = row[1:]
            number = tuple([int(x) for x in row[1:]])
            numberAndEvaluation = (number, isRequestedNumber)
            trainDataSet.append(numberAndEvaluation)
            # print "number and evaluation", numberAndEvaluation

    finally:
        f.close()

    return trainDataSet

def testReal(requestedNumber, perceptron):
    correctResultCount = 0

    f = open("csv/train.csv", 'r')
    correcness = 0
    wrongPredicted = 0
    missed = 0

    try:
        reader = csv.reader(f)
        line = 0
        for row in reader:
            if line == 0:
                line +=1
                continue
            # print "row", row
            realNumber = int(row[0])

            row = row[1:]
            number = tuple([int(x) for x in row[1:]])

            isRequestedNumber = realNumber == requestedNumber

            perceptronTest = perceptron.test(number)

            # print "perceptronTest :", perceptronTest, "request number :", realNumber

            if (perceptronTest and isRequestedNumber):
                correcness += 1

            if (perceptronTest and not isRequestedNumber):
                wrongPredicted += 1

            if (not perceptronTest and isRequestedNumber):
                missed += 1

    finally:
        f.close()

    print "correcness :", correcness, "wrongPredicted :", wrongPredicted, "missed :", missed
    return


def main():
    # example of perceptron creation with meta data
    dimension = 785
    threshold = 5
    learningRate = 0.01
    perceptron = pc.Perceptron(dimension, threshold, learningRate)

    print "Begin creating dataset"

    isThreeDataSet = createRealDataSet(3)

    print "Finish creating dataset"
    print "Begin training"

    perceptron.train(isThreeDataSet, 20)

    print "Finish training"

    print "Saving perceptron"
    # perceptron.saveWeights("isThree1")


    print "Begin Testing"
    # perceptron = pc.Perceptron.perceptronWithFilename("isThree1")
    #
    testReal(3, perceptron)

    print "Finish Testing"

main()
