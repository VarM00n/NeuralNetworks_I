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

    training_inputs = [[1, 0, 1, 0, 1, 0, 1, 0, 1],
                       [0, 1, 0, 1, 0, 1, 0, 1, 0],
                       [0, 0, 0, 1, 1, 1, 0, 0, 0],
                       [0, 1, 0, 0, 1, 0, 0, 1, 0],

                       [1, 0, 1, 0, 1, 0, 1, 0, 1],
                       [1, 0, 1, 0, 1, 0, 1, 0, 1],
                       [1, 0, 1, 0, 1, 0, 1, 0, 1],
                       [1, 0, 1, 0, 1, 0, 1, 0, 1],
                       [1, 0, 1, 0, 1, 0, 1, 0, 1],
                       [1, 0, 1, 0, 1, 0, 1, 0, 1]]

    training_outputs = [[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]]

    input_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                     0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                     0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                     0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    hidden_layer_weights = [0.5, 0.5, 0.5, 0.5,
                            0.5, 0.5, 0.5, 0.5,
                            0.5, 0.5, 0.5, 0.5,
                            0.5, 0.5, 0.5, 0.5]


perceptron = Perceptron()
for i in range(0, 1000):
    for j in range(0, len(perceptron.training_inputs)):
        # using algorithm presented on this site http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
        # forward
        hidden_layer_values = [0, 0, 0, 0]
        output_layer_values = [0, 0, 0, 0]
        training_input = perceptron.training_inputs[j]
        for k in range(0, len(hidden_layer_values)):
            sum_to_check = 0
            for m in range(0, len(training_input)):
                sum_to_check += training_input[m] * perceptron.input_weights[k * 9 + m]
            hidden_layer_values[k] = perceptron.sigmoid_function(sum_to_check)

        for k in range(0, len(perceptron.training_outputs[0])):
            sum_to_check = 0
            for m in range(0, len(output_layer_values)):
                sum_to_check += hidden_layer_values[m] * perceptron.hidden_layer_weights[k * 4 + m]
            output_layer_values[k] = perceptron.sigmoid_function(sum_to_check)

        # backward
        sigma2 = [0, 0, 0, 0]
        sigma = [0, 0, 0, 0]
        for k in range(0, len(output_layer_values)):
            sigma[k] = perceptron.training_outputs[j][k] - output_layer_values[k]
        for k in range(0, len(output_layer_values)):
            for m in range(0, len(output_layer_values)):
                sigma2[k] += sigma[m] * perceptron.hidden_layer_weights[k+m*len(output_layer_values)]

        # forward
        for k in range(0, len(output_layer_values)):
            for m in range(0, len(training_input)):
                perceptron.input_weights[m + k * len(training_input)] = \
                    perceptron.input_weights[m + k * len(training_input)] + sigma2[k] * \
                    perceptron.sigmoid_derivative(hidden_layer_values[k]) * hidden_layer_values[k]
        for k in range(0, len(output_layer_values)):
            for m in range(0, len(output_layer_values)):
                perceptron.hidden_layer_weights[m + k * len(output_layer_values)] = \
                    perceptron.hidden_layer_weights[m + k * len(output_layer_values)] + sigma[k] * \
                    perceptron.sigmoid_derivative(output_layer_values[k]) * output_layer_values[k]

print("yas")