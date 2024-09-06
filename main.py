import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_hidden = np.zeros((1, hidden_size))  # Bias for hidden layer

        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_output = np.zeros((1, output_size))  # Bias for output layer

        self.learning_rate = learning_rate

    # Forward pass
    def feedforward(self, X):
        # Hidden layer: input to hidden layer with bias
        self.hidden_layer_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_activation)

        # Output layer: hidden layer to output with bias
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        output = sigmoid(self.output_layer_activation)
        return output

    # Backward pass (backpropagation)
    def backpropagation(self, X, y, output):
        # Calculate the error at the output
        error = y - output
        d_output = error * sigmoid_derivative(output)

        # Calculate the error in the hidden layer
        error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases using the gradients
        self.weights_hidden_output += self.hidden_layer_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate

        self.weights_input_hidden += X.T.dot(d_hidden_layer) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate

    # Training the network
    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backpropagation(X, y, output)

# Prepare the training data
X_train = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
X_train_scaled = X_train / 10.0  # Scaling input

# Labels for even (1) and odd (0) numbers
y_train = np.array([[1], [0], [1], [0], [1], [0], [1], [0], [1], [0]])

# Initialize the neural network with more hidden neurons
nn = NeuralNetwork(input_size=1, hidden_size=10, output_size=1, learning_rate=0.1)

# Train the network for more epochs
nn.train(X_train_scaled, y_train, epochs=30000)

# Testing the network on new data (for reference, but neural network won't generalize well to new numbers)
test_data = np.array([[10], [11], [12], [13]])
test_data_scaled = test_data / 10.0  # Scaling test data

for test_case in test_data_scaled:
    result = nn.feedforward(test_case)
    classification = "parzysta" if result > 0.7 else "nieparzysta"
    print(f"[NN] Liczba: {int(test_case[0] * 10)}, Klasyfikacja: {classification}")

# -----------------------------------------------

# Simple Even-Odd Classifier using Modulo (Better approach for this task)

def classify_even_odd(numbers):
    for number in numbers:
        classification = "parzysta" if number % 2 == 0 else "nieparzysta"
        print(f"[Modulo] Liczba: {number}, Klasyfikacja: {classification}")

# Testing the classifier on numbers 10-13
test_data = np.array([10, 11, 12, 13])
classify_even_odd(test_data)
