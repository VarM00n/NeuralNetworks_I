import math


# Neural Network
class Perceptron:

    @staticmethod
    def sigmoid_function(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1-x)

    training_inputs = [[0, 0, 1],
                       [0, 1, 1],
                       [1, 0, 1],
                       [1, 0, 0]]
    training_outputs = [0, 0, 1, 1]
    weights = [0.5, 0.5, 0.5]


# Training network

perceptron = Perceptron()

for i in range(0, 10000):
    for j in range(0, len(perceptron.training_inputs)):
        training_input = perceptron.training_inputs[j]
        training_output = perceptron.training_outputs[j]

        sum_to_check = 0
        for k in range(0, len(training_input)):
            sum_to_check += training_input[k] * perceptron.weights[k]

        output = perceptron.sigmoid_function(sum_to_check)

        error = training_output - output

        adjustment = error * perceptron.sigmoid_derivative(output)

        for k in range(0, len(training_input)):
            perceptron.weights[k] += training_input[k] * adjustment

print("Network trained")

first_input = input()
second_input = input()
third_input = input()
user_input = [first_input, second_input, third_input]

sum_to_check1 = 0
for i in range(0, len(user_input)):
    sum_to_check1 += float(user_input[i]) * perceptron.weights[i]

user_output = perceptron.sigmoid_function(sum_to_check1)
print(user_output)

