import math
import random
import csv
import matplotlib.pyplot as plt

# import file_loading


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
    neurons_in_each_layer = [16, 15, 2]

    # attributes
    weights = []
    neurons = []

    x = []
    y = []

    training_inputs = []
    # xor, or, and
    training_outputs = []

    # constructor
    def __init__(self):
        self.generating_neurons()
        self.generating_weights()
        # self.new_training()
        self.training_network()
        plt.plot(self.y, self.x)
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.title('Error/epoch')
        plt.show()

    def generating_neurons(self):
        i = 0
        for layer in range(0, len(self.neurons_in_each_layer)):
            self.neurons.append([])
            for j in range(self.neurons_in_each_layer[layer]):
                self.neurons[i].append([0.3, 0])
            i += 1

    def generating_input(self, training_inputs, training_outputs):
        new_input = []
        new_output = []
        for digit in range(0, 16):  # tu trzeba bardziej zoptymalizować, żeby leciało po neurons_in_each_layer
            dig = random.uniform(0.3, 1)
            if dig > 0.5:
                new_input.append(1)
            else:
                new_input.append(0)
        # checking for vertical line in the middle
        if new_input[0] == 1 and new_input[5] == 1 and new_input[10] == 1 and new_input[15] == 1:
            new_output.append(1)
        else:
            new_output.append(0)
        if new_input[3] == 1 and new_input[6] == 1 and new_input[9] == 1 and new_input[12] == 1:
            new_output.append(1)
        else:
            new_output.append(0)
        training_inputs.append(new_input)
        training_outputs.append(new_output)


    def generating_weights(self):
        for layer in range(0, len(self.neurons_in_each_layer) - 1):
            self.weights.append([])
            for i in range(0, self.neurons_in_each_layer[layer + 1]):
                self.weights[layer].append([])
                for j in range(0, self.neurons_in_each_layer[layer]):
                    self.weights[layer][i].append(round_float(random.uniform(0, 1), 2))

    # def new_training(self):
    #     with open('train.csv', 'r') as read_obj:
    #         csv_reader = csv.reader(read_obj)
    #         count = 0
    #         for row in csv_reader:
    #             if count != 0:
    #                 self.training_outputs.append([])
    #                 self.training_inputs.append([])
    #                 for i in range(0, 10):
    #                     if int(row[0]) == i:
    #                         self.training_outputs[count - 1].append(1)
    #                     else:
    #                         self.training_outputs[count - 1].append(0)
    #                 for i in range(1, len(row)):
    #                     if int(row[i]) == 0:
    #                         self.training_inputs[count - 1].append(0)
    #                     else:
    #                         self.training_inputs[count - 1].append(1)
    #             count += 1
    #             # if count == 10:
    #             #     break
    #         print("done")

    def training_network(self):
        for r in range(0, 500):
            self.generating_input(self.training_inputs, self.training_outputs)
        # self.new_training()
        error_mean = 0
        for epoch in range(0, 200):
            sum_square = 0
            for training_input in range(len(self.training_inputs)):
                # print(training_input)
                # FORWARD PROPAGATION
                # assigning values from training input into neuron table
                self.assigning_training_values_to_network(training_input)
                # calculating total net input, squashing it using an activation function
                #   result is being saved into first value of neuron table
                self.forward_propagation()

                for k in range(0, len(self.training_outputs[training_input])):
                    sum_square += (self.training_outputs[training_input][k] - self.neurons[len(self.neurons_in_each_layer) - 1][k][0]) ** 2

                # backpropagation and weight changes
                # accessing every single weight in network
                self.backpropagation(training_input)
            self.x.append(sum_square)
            self.y.append(epoch)
            print("Error - " + str(sum_square))

    def assigning_training_values_to_network(self, training_input):
        for i in range(len(self.training_inputs[training_input])):
            self.neurons[0][i][0] = self.training_inputs[training_input][i]

    def forward_propagation(self):
        for layer in range(1, len(self.neurons_in_each_layer)):
            for neuron in range(len(self.neurons[layer])):
                net_input = 0
                for prev_neuron in range(0, len(self.neurons[layer - 1])):
                    net_input += self.neurons[layer - 1][prev_neuron][0] * \
                                 self.weights[layer - 1][neuron][prev_neuron]
                self.neurons[layer][neuron][0] = sigmoid_function(net_input)

    def backpropagation(self, training_input):
        for layer in range(len(self.weights) - 1, -1, -1):
            for weight_group in range(0, len(self.weights[layer])):
                for single_weight in range(0, len(self.weights[layer][weight_group])):
                    total_out = 0
                    # for last layer only
                    out = self.neurons[layer + 1][weight_group][0]
                    out_net = out * (1 - out)
                    if layer == len(self.weights) - 1:
                        total_out = - (self.training_outputs[training_input][weight_group] - out)
                        net_w = self.neurons[layer][single_weight][0]
                        self.neurons[layer + 1][weight_group][1] = total_out * out_net
                        self.weights[layer][weight_group][single_weight] -= 0.2 * total_out * out_net * net_w
                    # for every other layer
                    else:
                        for k in range(0, len(self.neurons[layer + 2])):
                            total_out += self.neurons[layer + 2][k][1] * self.weights[layer + 1][k][
                                weight_group]
                        net_w = self.neurons[layer][single_weight][0]
                        self.neurons[layer + 1][weight_group][1] = total_out * out_net
                        self.weights[layer][weight_group][single_weight] -= 0.2 * total_out * out_net * net_w


nn = NeuralNetwork()
print("Job done")

# values = []
#
# with open('test.csv', 'r') as read_obj:
#     csv_reader = csv.reader(read_obj)
#     count = 0
#     for row in csv_reader:
#         if count != 0:
#             values.append([])
#             for i in range(0, len(row)):
#                 if int(row[i]) == 0:
#                     values[count - 1].append(0)
#                 else:
#                     values[count - 1].append(1)
#         count += 1
#         if count == 12:
#             break
#     print("done")
#
# for i in range(0, 10):
#     for j in range(0, nn.neurons_in_each_layer[0]):
#         nn.neurons[0][j][0] = values[i][j]
#     for layer in range(1, len(nn.neurons_in_each_layer)):
#         for neuron in range(len(nn.neurons[layer])):
#             net_input = 0
#             for prev_neuron in range(0, len(nn.neurons[layer - 1])):
#                 net_input += float(nn.neurons[layer - 1][prev_neuron][0]) * \
#                              nn.weights[layer - 1][neuron][prev_neuron]
#             nn.neurons[layer][neuron][0] = sigmoid_function(net_input)
#     print(nn.neurons[2])
#
f = open("wages.txt", "a")
for layer in range(0, len(nn.weights)):
    for weight_group in range(0, len(nn.weights[layer])):
        f.write(str(layer) + "_" + str(weight_group) + "\n")
        for single_weight in range(0, len(nn.weights[layer][weight_group])):
            f.write(str(nn.weights[layer][weight_group][single_weight]) + "\n")

first_input = input()
# second_input = input()
# third_input = input()
# fourth_input = input()

# using function to generate test input and output
f = open("results.txt", "a")
input_for_nn = []
output_for_nn = []
for i in range(0, 50):
    nn.generating_input(input_for_nn, output_for_nn)

for i in range(0, 50):
    nn.neurons[0][0][0] = input_for_nn[i][0]
    nn.neurons[0][1][0] = input_for_nn[i][1]
    nn.neurons[0][2][0] = input_for_nn[i][2]
    nn.neurons[0][3][0] = input_for_nn[i][3]

    nn.neurons[0][4][0] = input_for_nn[i][4]
    nn.neurons[0][5][0] = input_for_nn[i][5]
    nn.neurons[0][6][0] = input_for_nn[i][6]
    nn.neurons[0][7][0] = input_for_nn[i][7]

    nn.neurons[0][8][0] = input_for_nn[i][8]
    nn.neurons[0][9][0] = input_for_nn[i][9]
    nn.neurons[0][10][0] = input_for_nn[i][10]
    nn.neurons[0][11][0] = input_for_nn[i][11]

    nn.neurons[0][12][0] = input_for_nn[i][12]
    nn.neurons[0][13][0] = input_for_nn[i][13]
    nn.neurons[0][14][0] = input_for_nn[i][14]
    nn.neurons[0][15][0] = input_for_nn[i][15]


    print("Wygląd wygenerowanego inputu :")
    print("\t" + str(input_for_nn[i][0]) + " " + str(input_for_nn[i][1]) + " " + str(input_for_nn[i][2]) + " " + str(input_for_nn[i][3]))
    print("\t" + str(input_for_nn[i][4]) + " " + str(input_for_nn[i][5]) + " " + str(input_for_nn[i][6]) + " " + str(input_for_nn[i][7]))
    print("\t" + str(input_for_nn[i][8]) + " " + str(input_for_nn[i][9]) + " " + str(input_for_nn[i][10]) + " " + str(input_for_nn[i][11]))
    print("\t" + str(input_for_nn[i][12]) + " " + str(input_for_nn[i][13]) + " " + str(input_for_nn[i][14]) + " " + str(input_for_nn[i][15]))

    f.write("Wygląd wygenerowanego inputu :" + "\n")
    f.write("\t" + str(input_for_nn[i][0]) + " " + str(input_for_nn[i][1]) + " " + str(input_for_nn[i][2]) + " " + str(input_for_nn[i][3]) + "\n")
    f.write("\t" + str(input_for_nn[i][4]) + " " + str(input_for_nn[i][5]) + " " + str(input_for_nn[i][6]) + " " + str(input_for_nn[i][7]) + "\n")
    f.write("\t" + str(input_for_nn[i][8]) + " " + str(input_for_nn[i][9]) + " " + str(input_for_nn[i][10]) + " " + str(input_for_nn[i][11]) + "\n")
    f.write("\t" + str(input_for_nn[i][12]) + " " + str(input_for_nn[i][13]) + " " + str(input_for_nn[i][14]) + " " + str(input_for_nn[i][15]) + "\n")

    for layer in range(1, len(nn.neurons_in_each_layer)):
        for neuron in range(len(nn.neurons[layer])):
            net_input = 0
            for prev_neuron in range(0, len(nn.neurons[layer - 1])):
                net_input += float(nn.neurons[layer - 1][prev_neuron][0]) * \
                             nn.weights[layer - 1][neuron][prev_neuron]
            nn.neurons[layer][neuron][0] = sigmoid_function(net_input)

    print("| -> " + str(nn.neurons[2][0][0]))
    print("- -> " + str(nn.neurons[2][1][0]))
    print("---------------------------------------------")
    f.write("| -> " + str(nn.neurons[2][0][0]) + "\n")
    f.write("- -> " + str(nn.neurons[2][1][0]) + "\n")
    f.write("---------------------------------------------" + "\n")
# print("or  - " + str(nn.neurons[2][0][0]))
# print("and - " + str(nn.neurons[2][1][0]))
