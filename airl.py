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

# Define the epsilon-greedy policy as a separate function
def choose_action(current_q, epsilon):
    """Epsilon-greedy policy for action selection."""
    if np.random.rand() < epsilon:
        return np.random.randint(0, len(current_q))
    else:
        return np.argmax(current_q)

# Define a simple environment for RL (a one-step environment)
def env_step(state, action):
    """
    A simple environment step function.
    The state is a vector of length 10.
    The "correct" action is determined by the first element of the state:
      - if state[0] < 0.33, correct action is 0
      - if 0.33 <= state[0] < 0.66, correct action is 1
      - if state[0] >= 0.66, correct action is 2
    Reward is 1.0 if the chosen action matches the correct action, else 0.0.
    A new random state is returned.
    """
    if state[0] < 0.33:
        correct_action = 0
    elif state[0] < 0.66:
        correct_action = 1
    else:
        correct_action = 2
    reward = 1.0 if action == correct_action else 0.0
    next_state = np.random.rand(10).astype(np.float32)
    done = True  # one-step episode
    return next_state, reward, done

# --- New: Function for one RL training step ---
def q_learning_step(state, layers_typed, l1, epsilon, gamma, learning_rate, manual_reward=None):
    """
    Performs one RL training step:
      1. Evaluates current Q-values.
      2. Chooses an action via epsilon-greedy policy.
      3. Steps through the environment.
      4. Optionally uses a manually supplied reward.
      5. Evaluates next state's Q-values.
      6. Computes target Q-values.
      7. Updates network weights using backpropagation.
    
    Returns:
      next_state, reward, done, action
    """
    # Evaluate current Q-values from output layer
    current_q = np.array([neuron.val for neuron in layers_typed[2]], dtype=np.float32)
    
    # Choose action using the epsilon-greedy policy
    action = choose_action(current_q, epsilon)
    
    # Take action in the environment, retrieving the auto reward
    next_state, auto_reward, done = env_step(state, action)
    
    # Use the manual reward if provided, otherwise use the auto reward
    reward = manual_reward if manual_reward is not None else auto_reward
    
    # Evaluate next state: set input layer values and perform forward pass
    for i, neuron in enumerate(l1):
        neuron.val = next_state[i]
    forward(layers_typed[0], layers_typed[1])
    forward(layers_typed[1], layers_typed[2])
    next_q = np.array([neuron.val for neuron in layers_typed[2]], dtype=np.float32)
    
    # Compute target Q-values
    target_q = current_q.copy()
    target_q[action] = reward + gamma * np.max(next_q) if not done else reward
    
    # Update network weights: restore current state and perform forward pass again
    for i, neuron in enumerate(l1):
        neuron.val = state[i]
    forward(layers_typed[0], layers_typed[1])
    forward(layers_typed[1], layers_typed[2])
    backward(layers_typed, target_q, learning_rate)
    
    return next_state, reward, done, action

"""
# Create layers (each layer is a typed list of Neuron objects)
l1 = make_layer(10)  # Input layer (state of size 10)
l2 = make_layer(10)   # Hidden layer
l3 = make_layer(3)   # Output layer (Q-values for 3 actions)

# Initialize weights between layers
make_weights(l1, l2)
make_weights(l2, l3)

# Create a typed list for all layers
layers_typed = TypedList()
layers_typed.append(l1)
layers_typed.append(l2)
layers_typed.append(l3)

# RL training parameters
num_episodes = 10000
learning_rate = 0.01
epsilon = 0.1     # Exploration probability
gamma = 0.9       # Discount factor

# Initialize state randomly
state = np.random.rand(10).astype(np.float32)

# Training loop using tqdm for progress bar
for episode in trange(num_episodes):
    for x in l1:
        x.val = ran.random()
    state, reward, done, action = q_learning_step(state, layers_typed, l1, epsilon, gamma, learning_rate, ran.random()*10)
    
# After training, print the final Q-values for the last state
print("Final Q-values for the last state:")
for i, neuron in enumerate(l3):
    print(f"Action {i}: Q-value = {neuron.val}")
"""