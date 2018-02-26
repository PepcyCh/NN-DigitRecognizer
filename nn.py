import neuralnetwork as net
import numpy as np
import json

with open('trainSet.json', 'r') as f:
    trainJsonDict = json.loads(f.read())

print '\033[1;33m[I]Finish reading\033[0m'

trainX = trainJsonDict['X']
trainY = trainJsonDict['Y']

print '\033[1;33m[I]Begin training\033[0m'

Net = net.NeuralNetwork(3, [784, 15, 10])
cost = Net.train(trainX, trainY)

print '\033[1;33m[I]Finish training.Cost = {}\033[0m'.format(cost)

Net.saveAsJson('digit_nn_theta.json')

with open('cvSet.json', 'r') as f:
    cvJsonDict = json.loads(f.read())

cvX = cvJsonDict['X']
cvY = cvJsonDict['Y']

print '\033[1;33m[I]Begin testing\033[0m'

predictRes = Net.predict(cvX, cvY)

print '\033[1;34mCorrect rate = {0}, cost = {1}\033[0m'.format(predictRes['correctRate'], predictRes['cost'])
