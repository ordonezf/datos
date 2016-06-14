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

def main():
    #example of perceptron creation with meta data
    dimension = 785
    threshold = 0.3
    learningRate = 0.1
    perceptron = pc.Perceptron(dimension, threshold, learningRate)

    isThreeDataSet = createRealDataSet(3)

    print isThreeDataSet

    perceptron.train(isThreeDataSet, 1000)

    perceptron.saveWeights("isThree")

    #training with sets as [((1, 2, 3, 0, 0, 0), 1), ((0, 0, 0, 3, 3, 8), 0)]

main()
