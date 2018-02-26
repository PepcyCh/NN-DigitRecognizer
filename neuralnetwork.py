import numpy as np
import json

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork(object):
    def __init__(self, layours, units, Epsilon=1):
        self.init(layours, units, Epsilon)

    def init(self, layours, units, Epsilon=1):
        self.LAYOURS_COUNT = layours
        self.UNITS_COUNT = units
        self.a = [None] * self.LAYOURS_COUNT
        self.z = [None] * self.LAYOURS_COUNT
        self.Theta = [None] * self.LAYOURS_COUNT
        self.delta = [None] * self.LAYOURS_COUNT
        self.Delta = [None] * self.LAYOURS_COUNT
        self.gradient = [None] * self.LAYOURS_COUNT

        for i in range(0, self.LAYOURS_COUNT - 1):
            self.Theta[i] = np.random.random( (self.UNITS_COUNT[i + 1], self.UNITS_COUNT[i] + 1) ) * 2 * Epsilon - Epsilon

    def frontPropagration(self, X):
        '''
        z^{(l)} = \Theta^{(l - 1)} * a^{(l - 1)}
                  | 1            |
        a^{(l)} = | Sig(z^{(l)}) |
        '''
    
        self.a[0] = X
        for i in range(1, self.LAYOURS_COUNT):
            self.z[i] = np.dot(self.Theta[i - 1], self.a[i - 1])
            self.a[i] = np.row_stack( (np.ones(X.shape[1]), sigmoid(self.z[i])) )
    
    def cost(self, H, Y, Lambda=0):
        '''
        H = h_{\Theta}(X)
        regularization = \frac{\lambda}{2m} \sum_{l = 1}^{L - 1} \sum_{i = 1}^{s_l} \sum_{j = 1}^{s_{l + 1}} (\Theta_{ji}^{(l)})^2
        J(\Theta) = \frac{1}{m} (- Y * \ln(H) - (1 - Y) * \ln(1 - H) + regularization)
        '''
    
        regularization = 0
        for i in range(0, self.LAYOURS_COUNT - 1):
            regularization += np.sum(self.Theta[i] ** 2)
    
        return (1.0 / Y.shape[1]) * \
            (-np.dot(Y, np.log(H).T)[0, 0] - np.dot(1.0 - Y, np.log(1.0 - H).T)[0, 0] + 0.5 * Lambda * regularization)

    def backPropagration(self, Y, Lambda):
        '''
        \delta^{(L)} = H - Y
        \delta^{(l)} = (\Theta^{(l)})^T * \delta^{(l + 1)} .* Sig'(z^{(l)})
        \Delta^{(l)} = \delta^{(l + 1)} * (a^{(l)})^T
        regularization = \lambda * \Theta^{(l)} * E' (E' := eye[0, 0] = 0)
        D^{(l)} = \frac{1}{m} (\Delta^{(l)} + regularization)
        '''

        self.delta[-1] = self.a[-1][1:] - Y
        self.Delta[-2] = np.dot(self.delta[-1], self.a[-2].T)
        self.gradient[-2] = (1.0 / Y.shape[1]) * \
                (self.Delta[-2] + Lambda * np.column_stack( (np.zeros(self.UNITS_COUNT[-1]), self.Theta[-2][:, 1:]) ))
        for i in range(self.LAYOURS_COUNT - 2, 0, -1):
            self.delta[i] = np.dot(self.Theta[i][:, 1:].T, self.delta[i + 1]) * dSigmoid(self.z[i])
            self.Delta[i - 1] = np.dot(self.delta[i], self.a[i - 1].T)
            self.gradient[i - 1] = (1.0 / Y.shape[1]) * \
                (self.Delta[i - 1] + Lambda * np.column_stack( (np.zeros(self.UNITS_COUNT[i]), self.Theta[i - 1][:, 1:]) ))

    def gradientDescent(self, X, Y, EPS=1e-8, Alpha=0.4, Lambda=0.001):
        err = 1.0
    
        self.frontPropagration(X)
        J = self.cost(self.a[-1][1:], Y, Lambda)
        while err > EPS:
            self.backPropagration(Y, Lambda)
            for i in range(0, self.LAYOURS_COUNT - 1):
                self.Theta[i] -= Alpha * self.gradient[i]

            self.frontPropagration(X)
            newJ = self.cost(self.a[-1][1:], Y, Lambda)
            err = J - newJ
            J = newJ

        return J

    def train(self, trainX, trainY, EPS=1e-7, Alpha=0.3, Lambda=0.001):
        X = np.column_stack( (np.ones(len(trainX)), np.array(trainX)) ).T
        Y = np.array(trainY).T

        return self.gradientDescent(X, Y, EPS, Alpha, Lambda)

    def predict(self, predictX, answerY=None):
        X = np.column_stack( (np.ones(len(predictX)), np.array(predictX)) ).T
        self.frontPropagration(X)

        prediction = self.a[-1][1:].T

        costVal = None
        if answerY is not None:
            costVal = self.cost(self.a[-1][1:], np.array(answerY).T)

        correctRate = None
        correctCount = 0
        for i in range(0, prediction.shape[0]):
            each = prediction[i]
            maxVal = 0.
            maxIndex = 0
            for j in range(0, each.size):
                if each[j] > maxVal:
                    maxVal = each[j]
                    maxIndex = j

            for j in range(0, each.size):
                if j == maxIndex:
                    each[j] = 1
                    if answerY is not None and answerY[i][j] == 1:
                        correctCount += 1
                else:
                    each[j] = 0
                    if answerY is not None and len(answerY[i]) == 1 and answerY[i][j] == 0:
                        correctCount += 1

        if answerY is not None:
            correctRate = float(correctCount) / len(answerY)

        return {'prediction': prediction, 'correctRate': correctRate, 'cost': costVal}

    def saveAsJson(self, filename='nn_theta.json'):
        layours = self.LAYOURS_COUNT
        units = self.UNITS_COUNT
        theta = []
        for i in self.Theta:
            if i is not None:
                theta.append(i.tolist())
            else:
                theta.append(None)

        jsonString = json.dumps({'layours': layours, 'units': units, 'theta': theta}, indent=4)
        # print jsonString
        with open(filename, 'w') as f:
            f.write(jsonString)

    def readFromJson(self, filename='nn_theta.json'):
        try:
            f = open(filename, 'r')
        except FileNotFoundError:
            print '{} doesn\'t exist.'
            return

        jsonDict = json.loads(f.read())
        self.LAYOURS_COUNT = jsonDict['layours']
        self.UNITS_COUNT = jsonDict['units']
        self.Theta = jsonDict['theta']
        for i in range(0, len(self.Theta)):
            if self.Theta[i] is not None:
                self.Theta[i] = np.array(self.Theta[i])
