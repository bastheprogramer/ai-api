import numpy as np
import random as ran
from numba import njit, float32, int32
from numba.experimental import jitclass
from numba.typed import List as TypedList
from tqdm import trange

# Define the structure for Numba JIT compilation
spec = [
    ('bias', float32),
    ('val', float32),
    ('delta', float32),
    ('weights', float32[:])  # NumPy array for weights
]

@jitclass(spec)
class Neuron:
    def __init__(self):
        self.bias = np.random.rand(1).astype(np.float32)[0]  # Initialize bias
        self.val = 0.0
        self.delta = 0.0
        self.weights = np.empty(0, dtype=np.float32)  # Start with an empty array

# Function to create a layer of neurons as a typed list
def make_layer(n):
    output = TypedList()
    for _ in range(n):
        output.append(Neuron())
    return output  # <-- IMPORTANT: return the typed list!

# Function to initialize weights between layers
def make_weights(prev_layer, next_layer):
    for neuron in next_layer:
        neuron.weights = np.random.rand(len(prev_layer)).astype(np.float32)

@njit
def relu(neurons):
    """Applies ReLU activation function."""
    for neuron in neurons:
        if neuron.val < 0.0:
            neuron.val = 0.0

@njit
def relu_derivative(neuron):
    """Derivative of ReLU function."""
    return 1.0 if neuron.val > 0.0 else 0.0

@njit
def forward(prev_layer, next_layer):
    """Forward pass computation."""
    for neuron in next_layer:
        total = 0.0
        # Use index-based iteration over the previous layer
        for i in range(len(prev_layer)):
            total += neuron.weights[i] * prev_layer[i].val
        neuron.val = total + neuron.bias
    relu(next_layer)

@njit
def backward(layers, targets, learning_rate):
    """
    Backpropagation algorithm to compute gradients and update weights.
    `layers` is expected to be a typed list of typed lists of Neuron objects.
    """
    num_layers = len(layers)
    # Compute error for output layer
    output_layer = layers[num_layers - 1]
    for i in range(len(output_layer)):
        neuron = output_layer[i]
        error = targets[i] - neuron.val
        neuron.delta = error * relu_derivative(neuron)
    # Backpropagate error through hidden layers
    for layer_idx in range(num_layers - 2, -1, -1):
        current_layer = layers[layer_idx]
        next_layer = layers[layer_idx + 1]
        for i in range(len(current_layer)):
            neuron = current_layer[i]
            neuron.delta = 0.0
            for j in range(len(next_layer)):
                neuron.delta += next_layer[j].weights[i] * next_layer[j].delta
            neuron.delta *= relu_derivative(neuron)
    # Update weights and biases
    for layer_idx in range(num_layers - 1):
        current_layer = layers[layer_idx]
        next_layer = layers[layer_idx + 1]
        for j in range(len(next_layer)):
            next_neuron = next_layer[j]
            for i in range(len(current_layer)):
                next_neuron.weights[i] += learning_rate * next_neuron.delta * current_layer[i].val
            next_neuron.bias += learning_rate * next_neuron.delta

# Create layers (each layer is a typed list of Neuron objects)
l1 = make_layer(10)  # Input layer
l2 = make_layer(10)   # Hidden layer
l3 = make_layer(3)   # Output layer

# Initialize weights between layers
make_weights(l1, l2)
make_weights(l2, l3)

# Create a typed list for all layers
layers_typed = TypedList()
layers_typed.append(l1)
layers_typed.append(l2)
layers_typed.append(l3)

# Set random input values for the input layer
for neuron in l1:
    neuron.val = np.random.rand(1).astype(np.float32)[0]

# Target output values
targets = np.array([0.2, 0.5, 0.8], dtype=np.float32)

# Training parameters
epochs = 10000
learning_rate = 0.01

# Training loop using tqdm for progress bar
for epoch in trange(epochs):
    # Forward pass: propagate inputs forward through the network
    for neuron in l1:
        neuron.val = np.random.rand(1).astype(np.float32)[0]
    forward(layers_typed[0], layers_typed[1])
    forward(layers_typed[1], layers_typed[2])
    # Backward pass: update weights and biases via backpropagation
    backward(layers_typed, targets, learning_rate)

# After training, print the final output layer values

for neuron in l1:
        neuron.val = np.random.rand(1).astype(np.float32)[0]
forward(layers_typed[0], layers_typed[1])
forward(layers_typed[1], layers_typed[2])
final_outputs = [neuron.val for neuron in layers_typed[2]]
print("Final output layer values:")
print(final_outputs)
