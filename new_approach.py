import math
import random


# rounding method
def round_float(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


# sigmoid function and its derivative
def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Neural Network
class NeuralNetwork:
    # Information about amount of layers and amount of neurons in each
    # eg. [20, 5, 4] - 3 layers.
    #   1st layer - 20 neurons
    #   2nd layer - 5 neurons
    #   3rd layer - 4 neurons
    # !!! Important: There has to be at least 2 layers (input, output) with at least one neuron each (perceptron)
    neutrons_in_each_layer = [20, 5, 4]

    # attributes
    weights = []
    neurons = []

    # constructor
    def __init__(self):
        self.generating_neurons()
        self.generating_weights()

    def generating_neurons(self):
        i = 0
        for layer in range(1, len(self.neutrons_in_each_layer)):
            self.neurons.append([])
            print(layer)
            for j in range(self.neutrons_in_each_layer[layer]):
                self.neurons[i].append([0, 0])
            i += 1

    def generating_weights(self):
        for layer in range(0, len(self.neutrons_in_each_layer) - 1):
            self.weights.append([])
            for i in range(0, self.neutrons_in_each_layer[layer + 1]):
                self.weights[layer].append([])
                for j in range(0, self.neutrons_in_each_layer[layer]):
                    self.weights[layer][i].append(round_float(random.uniform(0, 1), 2))


nn = NeuralNetwork()
print("Job done")
