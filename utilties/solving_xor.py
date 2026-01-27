import numpy as np
import tensorflow as tf
from keras import datasets


# Clean Coding Machine Learning, need to learn
# --- 1. Activation Functions ---
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    s = sigmoid(x)

    return s * (1 - s)


def relu(x):
    """
    Relu activation function
    Args:
        x (np.ndarray): input data.

    Returns: rectified linear activation function of x

    """
    return np.maximum(0, x)


def relu_derivative(x):
    """

    Args:
        x:

    Returns: derivative of relu(x)

    """
    return np.where(x > 0, 1, 0)


# --- 2. Neural Network Class ---
class SimpleNeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=0.1,
        activation_function="sigmoid",
    ):
        """
        Initializes the neural network with random weights and biases.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of neurons in the output layer.
            learning_rate (float): The step size for weight updates.
            activation_function: Activation Function Specified by the user
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_name = activation_function
        if self.activation_name == "sigmoid":
            self.activation_function = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            self.activation_function = relu
            self.activation_derivative = relu_derivative

        # Initialize weights and biases
        # Weights from input layer to hidden layer (W_ih)
        self.W_ih = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b_h = np.zeros((1, self.hidden_size))

        # Weights from hidden layer to output layer (W_ho)
        self.W_ho = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b_o = np.zeros((1, self.output_size))

    def forward(self, X):
        """
        Performs the forward pass through the network.

        Args:
            X (np.ndarray): Input data (batch_size, input_size).

        Returns:
            np.ndarray: Output predictions (batch_size, output_size).
        """
        # Input to hidden layer
        self.hidden_layer_input = np.dot(X, self.W_ih) + self.b_h
        self.hidden_layer_output = self.activation_function(self.hidden_layer_input)

        # Hidden to output layer
        self.output_layer_input = np.dot(self.hidden_layer_output, self.W_ho) + self.b_o
        self.predictions = sigmoid(self.output_layer_input)
        # print(f"Predictions: {self.predictions}")
        return self.predictions

    def backward(self, X, y, predictions):
        """
        Performs the backward pass (backpropagation) to compute gradients.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True target labels.
            predictions (np.ndarray): Predicted outputs from the forward pass.
        """
        # Calculate output layer error and delta
        output_error = y - predictions
        output_delta = output_error * sigmoid_derivative(self.output_layer_input)

        # Calculate hidden layer error and delta
        hidden_error = np.dot(output_delta, self.W_ho.T)
        hidden_delta = hidden_error * self.activation_derivative(
            self.hidden_layer_input
        )

        # Calculate gradients for weights and biases
        self.dW_ho = np.dot(self.hidden_layer_output.T, output_delta)
        self.db_o = np.sum(output_delta, axis=0, keepdims=True)

        self.dW_ih = np.dot(X.T, hidden_delta)
        self.db_h = np.sum(hidden_delta, axis=0, keepdims=True)

    def update_weights(self):  # no idea right now
        """Updates weights and biases using the calculated gradients."""
        self.W_ho += self.learning_rate * self.dW_ho
        self.b_o += self.learning_rate * self.db_o

        self.W_ih += self.learning_rate * self.dW_ih
        self.b_h += self.learning_rate * self.db_h

    def train(self, X, y, epochs):
        """
        Trains the neural network.

        Args:
            X (np.ndarray): Training input data.
            y (np.ndarray): True training labels.
            epochs (int): Number of training iterations.
        """
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)

            # Calculate loss (Mean Squared Error)
            loss = np.mean(np.square(y - predictions))

            # Backward pass (compute gradients)
            self.backward(X, y, predictions)

            # Update weights
            self.update_weights()

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Makes predictions on new input data.

        Args:
            X (np.ndarray): New input data.

        Returns:
            np.ndarray: Predicted outputs.
        """
        return self.forward(X)

    """
    def relu(x):
    '''ReLU activation function.'''
    return np.maximum(0, x)

def relu_derivative(x):
    '''Derivative of the ReLU function.'''
    return np.where(x > 0, 1, 0)

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, activation='sigmoid'):
        # ... (rest of your __init__)
        self.activation_name = activation
        if self.activation_name == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif self.activation_name == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        # ...
        self.hidden_layer_output = self.activation(self.hidden_layer_input)
        # ...
        self.predictions = sigmoid(self.output_layer_input) # Output layer often uses sigmoid for binary
        return self.predictions

    def backward(self, X, y, predictions):
        # ...
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_layer_input)
        # ...

    """


# --- 3. Example Usage ---


def main():
    # Define a simple dataset (e.g., for an OR-like gate or a simple classification)
    # X: Input features (batch_size, input_size)
    # y: Target labels (batch_size, output_size)

    # Dataset for a simple AND gate (modified slightly for demonstration)

    # --- 4. Test the Trained Network ---

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    input_size = X.shape[1]
    hidden_size = 4  # Number of neurons in the hidden layer
    output_size = y.shape[1]
    learning_rate = 0.01
    epochs = 50000

    print("--- Training a Simple Neural Network with NumPy ---")
    print(
        f"Input size: {input_size}, Hidden size: {hidden_size}, Output size: {output_size}"
    )
    print(f"Learning Rate: {learning_rate}, Epochs: {epochs}")

    # Initialize the neural network
    nn = SimpleNeuralNetwork(
        input_size, hidden_size, output_size, learning_rate, "relu"
    )

    # Train the network
    nn.train(X, y, epochs)
    print("\n--- Testing the Trained Network ---")

    # Test cases
    test_x = np.array([[0, 0]])
    test_y = np.array([1])
    predictions = nn.predict(test_x)
    print("\nOriginal Input (X):")
    print(test_x)
    print("\nPredicted Output (Raw Sigmoid):")
    print(predictions)

    print("\nPredicted Output (Rounded to 0 or 1):")
    print(np.round(predictions))

    print("\nTrue Output (y):")
    print(y)

    # Calculate final accuracy
    accuracy = np.mean(np.round(predictions[: len(y)]) == y)
    test_accuracy = np.mean(np.round(predictions[: len(test_y)]) == test_y)
    print(f"\nTraining Set Accuracy: {accuracy * 100:.2f}%")
    print(f"\nTest Set Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
