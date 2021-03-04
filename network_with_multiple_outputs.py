import math

# https://towardsdatascience.com/inroduction-to-neural-networks-in-python-7e0b422e6c24
# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/


# Neural Network
class Perceptron:

    @staticmethod
    def sigmoid_function(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1-x)

    amount_of_inputs = 4
    amount_of_neurons_in_hl = 3
    amount_of_outputs = 2

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
                       [0, 1, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 0]]

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
                        [0, 0],
                        [1, 0]]

    input_weights = [0.5, 0.5, 0.5, 0.5,
                     0.5, 0.5, 0.5, 0.5,
                     0.5, 0.5, 0.5, 0.5]

    hidden_layer_weights = [0.5, 0.5, 0.5,
                            0.5, 0.5, 0.5]


perceptron = Perceptron()

for z in range(0, 15000):
    for i in range(0, 16):
        hidden_layer_values = []
        output_layer_values = []
        sigma_for_output = []
        sigma_for_hl =[]
        # creating y's and sigmas
        for nhf in range(0, perceptron.amount_of_neurons_in_hl):
            hidden_layer_values.append(0)
            sigma_for_hl.append(0)
        for nol in range(0, perceptron.amount_of_outputs):
            output_layer_values.append(0)
            sigma_for_output.append(0)

        # forward propagate (input - hidden layer)
        for m in range(0, perceptron.amount_of_neurons_in_hl):
            sum_of_multiplication = 0
            for k in range(0, perceptron.amount_of_inputs):
                sum_of_multiplication += perceptron.training_inputs[i][k] * \
                                         perceptron.input_weights[k + m * perceptron.amount_of_inputs]
            hidden_layer_values[m] = perceptron.sigmoid_function(sum_of_multiplication)

        # forward propagate (hidden layer - output)
        for m in range(0, perceptron.amount_of_outputs):
            sum_of_multiplication = 0
            for k in range(0, perceptron.amount_of_neurons_in_hl):
                sum_of_multiplication += hidden_layer_values[k] * \
                                         perceptron.hidden_layer_weights[k + m * perceptron.amount_of_neurons_in_hl]
            output_layer_values[m] = perceptron.sigmoid_function(sum_of_multiplication)

        # end of forward propagation

        # start of backpropagation
        # output
        for k in range(perceptron.amount_of_outputs):
            sigma_for_output[k] = perceptron.training_outputs[i][k] - output_layer_values[k]

        # hidden layer
        for k in range(perceptron.amount_of_neurons_in_hl):
            sum_for_hl = 0
            for m in range(perceptron.amount_of_outputs):
                sum_for_hl += output_layer_values[m] * \
                              perceptron.hidden_layer_weights[k + m * perceptron.amount_of_neurons_in_hl]
            sigma_for_hl[k] = sum_for_hl

        # forward again
        # hidden layer
        for k in range(perceptron.amount_of_neurons_in_hl):
            for m in range(perceptron.amount_of_inputs):
                perceptron.input_weights[m + k * perceptron.amount_of_inputs] += \
                    sigma_for_hl[k] * perceptron.sigmoid_derivative(hidden_layer_values[k]) * \
                    perceptron.training_inputs[i][m]

        # output layer
        for k in range(perceptron.amount_of_outputs):
            for m in range(perceptron.amount_of_neurons_in_hl):
                perceptron.hidden_layer_weights[m + k * perceptron.amount_of_neurons_in_hl] += \
                    sigma_for_output[k] * perceptron.sigmoid_derivative(output_layer_values[k]) * hidden_layer_values[k]

print("Network trained")

first_input = input()
second_input = input()
third_input = input()
fourth_input = input()
user_input = [first_input, second_input, third_input, fourth_input]
hidden_layer_values = [0, 0, 0]
output_layer_values = [0, 0]

for m in range(0, perceptron.amount_of_neurons_in_hl):
    sum_of_multiplication = 0
    for k in range(0, perceptron.amount_of_inputs):
        sum_of_multiplication += float(user_input[k]) * \
                                 perceptron.input_weights[k + m * perceptron.amount_of_inputs]
    hidden_layer_values[m] = perceptron.sigmoid_function(sum_of_multiplication)

# forward propagate (hidden layer - output)
for m in range(0, perceptron.amount_of_outputs):
    sum_of_multiplication = 0
    for k in range(0, perceptron.amount_of_neurons_in_hl):
        sum_of_multiplication += hidden_layer_values[k] * \
                                 perceptron.hidden_layer_weights[k + m * perceptron.amount_of_neurons_in_hl]
    output_layer_values[m] = perceptron.sigmoid_function(sum_of_multiplication)

print(output_layer_values)
