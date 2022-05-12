import math
import sys
from pprint import pprint

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPushButton, QWidget, QGridLayout, QApplication, QGroupBox, QSlider, QVBoxLayout, QLabel, \
    QFrame, QComboBox, QStyleFactory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

windowHeight = 1200
windowWidth = 900
# deviation = 0.125
#epochAmount = 50000
rng = np.random.default_rng()

np.set_printoptions(threshold=np.inf)

class NeuronLayer:
    learningRate = 0.001
    sigmoidBeta = -2

    def __init__(self, activationFunction, neuronAmount, trainingData, parent=None, child=None):
        self.parent = parent
        self.child = child
        self.trainingData = trainingData
        self.activationFunction = activationFunction
        self.neuronAmount = neuronAmount

    def sigmoid(self, value):
        return 1.0 / (1 + np.exp(self.sigmoidBeta * value))

    def ReluDerivative(self, value):
        return np.where(value > 0, 1, 0)

    def leakyReluDerivative(self, value):
        return np.where(value > 0, 1, 0.01)

    def leakyRelu(self, value):
        return np.where(value > 0, value, value * 0.01)

    def useActivationFunction(self, value):
        match self.activationFunction:
            case "heaviside":
                return np.heaviside(value, 0.5)
            case "sin":
                return np.sin(value)
            case "tanh":
                return np.tanh(value)
            case "sigmoid":
                return self.sigmoid(value)
            case "relu":
                return np.maximum(value, 0)
            case "leakyrelu":
                return self.leakyRelu(value)
            case "sign":
                return np.sign(value)
            case _:
                return 0

    def activationFunctionDerivative(self, value):
        match self.activationFunction:
            case "heaviside":
                return np.ones(value.shape)
            case "sin":
                return np.cos(value)
            case "tanh":
                return 1 - (np.tanh(value) * np.tanh(value))
            case "sigmoid":
                return self.sigmoid(value) * (1 - self.sigmoid(value))
            case "relu":
                return self.ReluDerivative(value)
            case "leakyrelu":
                return self.leakyReluDerivative(value)
            case "sign":
                return np.ones(value.shape)

    def generateBase(self):
        #pprint(self.trainingData)
        self.trainingData = np.c_[self.trainingData, np.ones(np.shape(self.trainingData)[0])]

    def calculateOutput(self): #4
        self.state = self.trainingData @ self.weights
        self.output = self.useActivationFunction(self.state)
        return self.output

    def hiddenInit(self, inputAmount, outputAmount, labels):
        self.generateWeights(inputAmount, outputAmount)
        self.setLabels(labels)
        self.calculateOutput()

    def setLabels(self, labels):
        self.labels = labels.reshape(-1, 1)

    def getLabels(self):
        return self.labels

    def calculateError(self):
        self.error = self.output - self.labels
        #self.error = pow(self.output - self.labels, 2)
        return self.error

    def generateWeights(self, inputAmount, outputAmount):
        self.weights = np.random.rand(inputAmount, outputAmount)

    # def generateWeights(self):
    #     self.weights = np.random.rand(len(self.trainingData[0]), self.neuronAmount)

    def updateWeights(self, weights):
        self.weights = weights

    def getParent(self):
        return self.parent

    def getChild(self):
        return self.child

    def setChild(self, child):
        self.child = child

    def setInput(self, input):
        self.trainingData = input

    def getLayerOutput(self):
        return self.output

    def getLayerInput(self):
        return self.trainingData

    def getTransposedData(self):
        return np.transpose(self.trainingData)

    def forward(self, x):
        #pprint(self.weights.shape)
        self.state = x @ self.weights
        self.output = self.useActivationFunction(self.state)
        self.derivativeStates = self.activationFunctionDerivative(self.state)
        return self.output

    # def forward(self):
    #     if self.child is not None:
    #         self.calculateOutput()
    #         self.calculateDerivativeState()
    #         self.child.setInput(self.output)
    #         self.child.forward()

    def calculateDerivativeState(self):
        self.derivativeStates = self.activationFunctionDerivative(self.state)

    def getBackData(self):
        return self.weights @ self.derivativeStates.T

    def getDelta(self):
        return self.delta

    def getWeights(self):
        return self.weights

    def printData(self):
        print("Weights: ")
        pprint(self.weights.shape)
        print("Input data: ")
        pprint(self.trainingData.shape)
        print("Transposed data: ")
        pprint(self.trainingData.T.shape)
        print("Output: ")
        pprint(self.output.shape)
        print("Derivative of state: ")
        pprint(self.derivativeStates.shape)
        print("Error (label - input): ")
        pprint(self.error.shape)
        print("Label: ")
        pprint(self.labels.shape)
        print("Data to pass: ")
        pprint(self.getBackData().shape)
        print("Learning rate: ")
        pprint(self.learningRate)

    def update(self, input):
        updateValue = input.T.dot(self.delta)
        self.weights -= self.learningRate * updateValue

    def calculateDeltaWeightOutput(self):
        self.delta = (self.error * self.derivativeStates)
        return self.delta

    def calculateDeltaWeight(self):
        self.delta = self.child.delta @ self.child.weights.T * self.derivativeStates
        return self.delta

    def backpropagate(self):
        if self.child is None:
            self.calculateError()
            self.calculateDerivativeState()
            self.calculateDeltaWeightOutput()
        else:
            self.calculateDerivativeState()
            self.calculateDeltaWeight()
        return self.delta

class NeuronNetwork:
    activationFunction = "heaviside"
    epochs = 1000
    inputAmount = 2
    outputAmount = 2

    def extractLabel(self, trainingData): #1
        rowLength = len(trainingData[0])
        labels = trainingData[:, rowLength - 1]
        trainingData = np.delete(trainingData, rowLength - 1, 1)
        return labels, trainingData

    def __init__(self, trainingData, activationFunction, layerAmount, neuronAmount):
        labels, trainingData = self.extractLabel(trainingData)
        self.trainingData = trainingData
        self.activationFunction = activationFunction
        self.layerAmount = layerAmount
        self.neuronAmount = neuronAmount
        self.layerList = [NeuronLayer(activationFunction, self.inputAmount, trainingData)]
        self.layerList[0].generateBase()
        length = len(trainingData[0])+1
        self.layerList[0].hiddenInit(length, self.inputAmount+1, labels)
        #self.layerList[0].hiddenInit(length, neuronAmount+1, labels)
        for i in range(1, layerAmount-1):
            #print(self.layerList[i-1].getLayerOutput().shape)
            self.layerList.append(NeuronLayer(activationFunction, neuronAmount, self.layerList[i-1].getLayerOutput(), self.layerList[i-1]))
            #print(len(self.layerList[i].trainingData[0]))
            #print(neuronAmount+1)
            self.layerList[i].hiddenInit(len(self.layerList[i].trainingData[0]), neuronAmount+1, labels)
            self.layerList[i-1].setChild(self.layerList[i])
            #pprint(self.layerList[i].weights.shape)

        self.layerList.append(NeuronLayer(activationFunction, self.outputAmount, self.layerList[layerAmount-2].getLayerOutput(), self.layerList[layerAmount-2]))
        self.layerList[layerAmount-1].hiddenInit(len(self.layerList[layerAmount-1].trainingData[0]), self.outputAmount, labels)
        self.layerList[layerAmount-2].setChild(self.layerList[layerAmount-1])

    def trainNetwork(self, epochs):
        trainInput = np.c_[self.trainingData, np.ones(np.shape(self.trainingData)[0])]
        for i in range(epochs):
            input = trainInput
            for layer in self.layerList:
                input = layer.forward(input)
            self.layerList[self.layerAmount-1].backpropagate()
            for i in reversed(range(self.layerAmount)):
                self.layerList[i].backpropagate()
            input = trainInput
            for layer in self.layerList:
                layer.update(input)
                input = layer.output
        #pprint(input)
            #self.layerList[0].forward()

        #test = np.c_[self.trainingData, np.ones(np.shape(self.trainingData)[0]) * -1]
        #pprint(test.shape)
        #return test

    def predict(self, input):
        bias = (np.ones(np.shape(input)[0]))
        input = np.c_[input, bias.reshape(-1, 1)]
        #pprint(input)
        for layer in self.layerList:
            input = layer.forward(input)
            #pprint(layer.weights)
            #pprint(input)
        #pprint(input)
        return input
        #for layer in self.layerList:


class PlotGroup(QWidget):
    def __init__(self, plot_name):
        super().__init__()
        self.name = plot_name
        self.groupBox = QGroupBox(self.name)
        self.vBox = QVBoxLayout()
        self.fig = plt.figure()
        self.plot = FigureCanvas(self.fig)
        self.vBox.addWidget(self.plot)
        self.groupBox.setLayout(self.vBox)
        self.visualize(1, 500, 0.05)
        # self.createNeuron("heaviside")

    def generateData(self, modes, samples, deviation):
        x = np.random.uniform(0, 1, modes)
        y = np.random.uniform(0, 1, modes)
        theta = np.random.uniform(0, deviation, modes)
        normal_x = np.random.normal(x[0], theta[0], samples)
        normal_y = np.random.normal(y[0], theta[0], samples)
        for i in range(1, modes):
            normal_x = np.concatenate([normal_x, np.random.normal(x[i], theta[i], samples)])
            normal_y = np.concatenate([normal_y, np.random.normal(y[i], theta[i], samples)])
        return np.c_[np.transpose(normal_x), np.transpose(normal_y)], np.c_[np.transpose(x), np.transpose(y)]

    def createNeuralNetwork(self, layerAmount, neuronAmount, activationFunction):
        self.neuronNetwork = NeuronNetwork(self.data, activationFunction, layerAmount, neuronAmount)

    def trainNeuralNetwork(self, epochAmount=1000):
        self.neuronNetwork.trainNetwork(epochAmount)
        xx, yy = self.prepareDecisionBoundary()
        input = np.c_[xx.reshape(-1, 1), yy.reshape(-1, 1)]
        #pprint(input.shape)
        zz = self.neuronNetwork.predict(input)
        #pprint(zz)
        zz = zz[:, 0].reshape(xx.shape)
        self.generateBoundary(zz, xx, yy)
        self.plot.draw()

    def visualize(self, modes, samples, deviation):
        if samples % 50 != 0:
            samples = math.floor(samples / 50) * 50
        self.samples = samples
        self.modes = modes
        self.deviation = deviation
        self.fig.clear()
        ax = self.fig.add_subplot()

        self.classOne, self.classOneOrigin = self.generateData(modes, samples, deviation)
        tempArray = np.ones([samples*modes, 1])
        classOneData = np.c_[self.classOne, tempArray]
        self.classTwo, self.classTwoOrigin = self.generateData(modes, samples, deviation)
        tempArray = np.ones([samples*modes, 1]) * 0
        classTwoData = np.c_[self.classTwo, tempArray]
        self.data = np.concatenate([classOneData, classTwoData])
        np.random.shuffle(self.data)
        self.drawSamples("red", "darkRed", "green", "darkGreen", ax)
        self.plot.draw()

    def drawSamples(self, colorOneSamples, colorOneMode, colorTwoSamples, colorTwoMode, ax):
        x = np.linspace(0, 1, 2)
        ax.scatter(x, x, label=type, alpha=0)
        ax.scatter(self.classOne[:, 0], self.classOne[:, 1], color=colorOneSamples, marker='D')
        ax.scatter(self.classTwo[:, 0], self.classTwo[:, 1], color=colorTwoSamples, marker='D')
        ax.scatter(self.classOneOrigin[:, 0], self.classOneOrigin[:, 1], color=colorOneMode, marker='x', s=100)
        ax.scatter(self.classTwoOrigin[:, 0], self.classTwoOrigin[:, 1], color=colorTwoMode, marker='x', s=100)

    def generateBoundary(self, prediction, xx, yy):
        self.fig.clear()
        ax = self.fig.add_subplot()
        ax.contourf(xx, yy, prediction)
        self.drawSamples("red", "darkRed", "green", "darkGreen", ax)

    def prepareDecisionBoundary(self):
        minX = np.min(np.append(self.data[:, 0], np.array([0])))
        maxX = np.max(np.append(self.data[:, 0], np.array([1])))
        minY = np.min(np.append(self.data[:, 1], np.array([0])))
        maxY = np.max(np.append(self.data[:, 1], np.array([1])))
        x = np.linspace(minX, maxX, 50)
        y = np.linspace(minY, maxY, 50)
        xx, yy = np.meshgrid(x, y)
        # print(xx)
        # print(yy)
        return xx, yy

    def getGroupBox(self):
        return self.groupBox


class ButtonGroup(QWidget):
    def __init__(self, group_name, buttons, buttonsPerRow=3):
        super().__init__()
        self.name = group_name
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.titleLabel = QLabel(group_name, self)
        self.titleLabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.titleLabel.setFont(QFont('Arial', 25, QFont.Bold))
        self.grid.addWidget(self.titleLabel, 0, 0, 1, 3)
        countX = 0
        countY = 1
        for i in buttons:
            self.grid.addWidget(i, countY, countX % buttonsPerRow)
            countX += 1
            if countX % buttonsPerRow == 0:
                countY += 1

    def getLayout(self):
        return self.grid


class DropboxGroup(QWidget):
    def __init__(self, group_name, options):
        super().__init__()
        self.name = group_name
        self.frame = QFrame()
        self.comboBox = QComboBox(self)
        for i in options:
            self.comboBox.addItem(i)
        self.titleLabel = QLabel(group_name, self)
        self.titleLabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.titleLabel.setFont(QFont('Arial', 25, QFont.Bold))
        self.vBox = QVBoxLayout()
        # self.vBox.addStretch()
        self.vBox.addWidget(self.titleLabel)
        self.vBox.addWidget(self.comboBox)
        # self.vBox.addSpacing(15)
        # self.vBox.addStretch()
        self.frame.setMinimumWidth(int(windowWidth / 3))
        self.frame.setLayout(self.vBox)

    def getFunction(self):
        return self.comboBox.currentText()

    def getFrame(self):
        return self.frame


class SliderGroup(QWidget):
    def __init__(self, group_name, minimum=0, maximum=16, step=1):
        super().__init__()
        self.name = group_name
        self.frame = QFrame()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setTickInterval(step)
        self.slider.setSingleStep(step)
        self.slider.setValue(int(maximum / 2))
        self.slider.valueChanged.connect(self.changeValue)

        self.titleLabel = QLabel(group_name, self)
        self.titleLabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.titleLabel.setFont(QFont('Arial', 25, QFont.Bold))
        self.valueLabel = QLabel(str(self.slider.value()), self)
        self.valueLabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.valueLabel.setFont(QFont('Arial', 20))
        self.changeValue(maximum / 2)

        self.vBox = QVBoxLayout()
        self.vBox.addStretch()
        self.vBox.addWidget(self.titleLabel)
        self.vBox.addWidget(self.slider)
        self.vBox.addSpacing(15)
        self.vBox.addWidget(self.valueLabel)
        self.vBox.addStretch()
        self.frame.setMinimumWidth(int(windowWidth / 3))
        self.frame.setLayout(self.vBox)

    def getName(self):
        return self.name

    def getFrame(self):
        return self.frame

    def changeValue(self, value):
        self.valueLabel.setText(str(value))

    def getValue(self):
        return self.slider.value()


class SliderDeviationGroup(SliderGroup):
    def changeValue(self, value):
        self.valueLabel.setText(str(round(value * 0.05, 2)))

    def getValue(self):
        return self.slider.value() * 0.05


class ShowcaseWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def sigmoid(self, value):
        return 1.0 / (1 + np.exp(-6 * value))

    def leakyRelu(self, value):
        return np.where(value > 0, value, value * 0.01)

    def useActivationFunction(self, value, type):
        match type:
            case "heaviside":
                return np.heaviside(value, 0.5)
            case "sin":
                return np.sin(value)
            case "tanh":
                return np.tanh(value)
            case "sigmoid":
                return self.sigmoid(value)
            case "relu":
                return np.maximum(value, 0)
            case "leakyrelu":
                return self.leakyRelu(value)
            case "sign":
                return np.sign(value)
            case _:
                return 0

    def drawPlot(self, type):
        figure = plt.figure()
        plot = FigureCanvas(figure)
        # x = np.linspace(-1, 1, 11)
        x = np.linspace(-1, 1, 1001)
        ax = figure.add_subplot()
        ax.scatter(x, x, label=type, alpha=0)
        y = self.useActivationFunction(x, type)
        # plt.plot(x, y)
        ax.scatter(x, y, s=5)
        plt.title(type)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        plt.grid(True)
        plot.draw()
        return plot

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)
        self.move(300, 150)
        self.resize(int(windowHeight), int(windowWidth))
        self.setWindowTitle('Showcase')

        grid.addWidget(self.drawPlot("heaviside"), 0, 0)
        grid.addWidget(self.drawPlot("sin"), 0, 1)
        grid.addWidget(self.drawPlot("tanh"), 0, 2)
        grid.addWidget(self.drawPlot("sigmoid"), 1, 0)
        grid.addWidget(self.drawPlot("relu"), 1, 1)
        grid.addWidget(self.drawPlot("leakyrelu"), 1, 2)
        grid.addWidget(self.drawPlot("sign"), 2, 0)

    def showWindow(self):
        self.show()


class Exercise(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def visualizePlot(self):
        #self.plot.visualize(1, self.samples.getValue(), 0.05)
        self.plot.visualize(self.modes.getValue(), self.samples.getValue(), 0.05)

        # self.plot.visualize(self.modes.getValue(), self.samples.getValue(), self.deviation.getValue())

    def createNeuralNetwork(self):
        self.plot.createNeuralNetwork(self.layers.getValue(), self.neuronsPerLayer.getValue(),
                                      self.dropbox.getFunction())

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)
        self.plot = PlotGroup("Plot")
        grid.addWidget(self.plot.getGroupBox(), 0, 0, 4, 1)
        activationFunctions = ['heaviside', 'sigmoid', 'sin', 'tanh', 'sign', 'relu', 'leakyrelu']
        self.dropbox = DropboxGroup("Activation", activationFunctions)
        grid.addWidget(self.dropbox.getFrame(), 0, 1)

        #createButton = QPushButton('Create')
        #trainButton = QPushButton('Train')
        showcaseButton = QPushButton('Showcase')
        createNetworkButton = QPushButton('Create network')
        trainNetworkButton = QPushButton('Train network')
        buttonArray = [showcaseButton, createNetworkButton, trainNetworkButton]

        self.buttons = ButtonGroup("Neuron", buttonArray)
        grid.addWidget(self.buttons, 0, 2)

        self.showcase = ShowcaseWindow()
        #createButton.clicked.connect(lambda: self.createNeuron())
        #trainButton.clicked.connect(lambda: self.plot.trainNeuron())

        self.modes = SliderGroup("Modes per class", 1, 10)
        grid.addWidget(self.modes.getFrame(), 1, 1)

        self.samples = SliderGroup("Samples per mode", 50, 1000, 50)
        grid.addWidget(self.samples.getFrame(), 1, 2)

        self.layers = SliderGroup("Layer amount", 3, 6, 1)
        self.neuronsPerLayer = SliderGroup("Neurons per layer", 2, 10, 1)

        grid.addWidget(self.layers.getFrame(), 2, 1)
        grid.addWidget(self.neuronsPerLayer.getFrame(), 2, 2)

        self.epochs = SliderGroup("Epoch amount", 10000, 200000, 10000)

        grid.addWidget(self.epochs.getFrame(), 3, 1, 1, 2)

        button = QPushButton('Visualize')
        button.setFont(QFont('Arial', 25, QFont.Bold))
        button.setFixedHeight(int(windowHeight / 8))
        button.clicked.connect(lambda: self.visualizePlot())
        showcaseButton.clicked.connect(lambda: self.showcase.show())
        createNetworkButton.clicked.connect(lambda: self.createNeuralNetwork())
        trainNetworkButton.clicked.connect(lambda: self.plot.trainNeuralNetwork(self.epochs.getValue()))
        grid.addWidget(button, 4, 0, 1, 3)
        self.move(300, 150)
        self.resize(windowHeight, windowWidth)
        self.setWindowTitle('Data generator')
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    ex = Exercise()
    sys.exit(app.exec_())
