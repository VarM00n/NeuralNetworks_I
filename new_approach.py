import math
import random


# W lewo i w góre - zwrost wiązania metalicznego
# W prawo i w dół - zwrost wiązania niemetalicznego


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
    neurons_in_each_layer = [4, 5, 2]

    # attributes
    weights = []
    neurons = []

    training_inputs = [[1, 1, 1, 1],
                       [1, 1, 1, 0],
                       [1, 1, 0, 1],
                       [1, 0, 1, 1],
                       [0, 1, 1, 1],
                       [1, 1, 0, 0],
                       [1, 0, 1, 0],
                       [0, 1, 1, 0],
                       [1, 0, 0, 1],
                       [0, 1, 0, 1],
                       [0, 0, 1, 1],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0],
                       [1, 0, 0, 0]]

    training_outputs = [[0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [1, 0],
                        [0, 0],
                        [0, 0],
                        [1, 0],
                        [0, 0],
                        [1, 0],
                        [1, 0],
                        [1, 0],
                        [1, 0],
                        [1, 0],
                        [1, 0]]

    # constructor
    def __init__(self):
        self.generating_neurons()
        self.generating_weights()
        self.training_network()

    def generating_neurons(self):
        i = 0
        for layer in range(0, len(self.neurons_in_each_layer)):
            self.neurons.append([])
            for j in range(self.neurons_in_each_layer[layer]):
                self.neurons[i].append([0, 0])
            i += 1

    def generating_weights(self):
        for layer in range(0, len(self.neurons_in_each_layer) - 1):
            self.weights.append([])
            for i in range(0, self.neurons_in_each_layer[layer + 1]):
                self.weights[layer].append([])
                for j in range(0, self.neurons_in_each_layer[layer]):
                    self.weights[layer][i].append(round_float(random.uniform(0, 1), 2))

    def training_network(self):
        for epoch in range(0, 1000):
            for training_input in range(len(self.training_inputs)):
                total_error = []
                # FORWARD PROPAGATION
                # assigning values from training input into neuron table
                for i in range(len(self.training_inputs[training_input])):
                    self.neurons[0][i][0] = self.training_inputs[training_input][i]
                # calculating total net input, squashing it using an activation function
                #   result is being saved into first value of neuron table
                for layer in range(1, len(self.neurons_in_each_layer)):
                    for neuron in range(len(self.neurons[layer])):
                        net_input = 0
                        for prev_neuron in range(0, len(self.neurons[layer - 1])):
                            net_input += self.neurons[layer - 1][prev_neuron][0] * \
                                         self.weights[layer - 1][neuron][prev_neuron]
                        self.neurons[layer][neuron][0] = sigmoid_function(net_input)
                # backpropagation and weight changes
                # accessing every single weight in network
                for layer in range(len(self.weights) - 1, -1, -1):
                    for weight_group in range(0, len(self.weights[layer])):
                        for single_weight in range(0, len(self.weights[layer][weight_group])):
                            out = self.neurons[layer + 1][weight_group][0]
                            out_net = out * (1 - out)
                            sum_of_total_out = 0
                            net_w = self.neurons[layer+1][weight_group][0]
                            if layer == len(self.weights) - 1:
                                total_out = -(self.training_outputs[training_input][weight_group] - out)
                                self.neurons[layer+1][weight_group][1] = total_out * out_net
                                self.weights[layer][weight_group][single_weight] -= total_out * out_net * net_w
                            else:
                                for i in range(0, len(self.neurons[layer+2])):
                                    sum_of_total_out += self.neurons[layer+2][i][1]
                                self.weights[layer][weight_group][single_weight] -= sum_of_total_out * out_net * net_w


nn = NeuralNetwork()
print("Job done")
# first_input = input()
# second_input = input()
# third_input = input()
# fourth_input = input()
#
# nn.neurons[0][0][0] = first_input
# nn.neurons[0][1][0] = second_input
# nn.neurons[0][2][0] = third_input
# nn.neurons[0][3][0] = fourth_input
#
# for layer in range(1, len(nn.neurons_in_each_layer)):
#     for neuron in range(len(nn.neurons[layer])):
#         net_input = 0
#         for prev_neuron in range(0, len(nn.neurons[layer - 1])):
#             net_input += float(nn.neurons[layer - 1][prev_neuron][0]) * \
#                          nn.weights[layer - 1][neuron][prev_neuron]
#         nn.neurons[layer][neuron][0] = sigmoid_function(net_input)
# print(nn.neurons[2][0][0])
# print(nn.neurons[2][1][0])
