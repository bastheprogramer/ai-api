import numpy as np
import random as ran
from numba import njit, float32, int32
from numba.experimental import jitclass
from numba.typed import List as TypedList
from tqdm import trange

# =========================
# NEURAL NETWORK DEFINITION
# =========================

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

# -------------------------
# Environment & RL Function
# -------------------------

def choose_action(current_q, epsilon):
    """Epsilon-greedy policy for action selection."""
    if np.random.rand() < epsilon:
        return np.random.randint(0, len(current_q))
    else:
        return np.argmax(current_q)

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

# -------------------------
# Genetic Algorithm (GA)
# -------------------------
# In this GA we will treat a network’s learnable parameters (the biases and weights
# in layers 2 and 3) as the “genome.” We define functions to (a) create a new network,
# (b) clone a network (deep copy), (c) perform crossover between two networks, (d)
# mutate a network, and (e) evaluate its performance in the environment.

def create_network():
    """Creates a new network with 3 layers: input (10 neurons), hidden (10 neurons), and output (3 neurons)."""
    l1 = make_layer(10)   # Input layer
    l2 = make_layer(10)   # Hidden layer
    l3 = make_layer(3)    # Output layer
    make_weights(l1, l2)
    make_weights(l2, l3)
    layers_typed = TypedList()
    layers_typed.append(l1)
    layers_typed.append(l2)
    layers_typed.append(l3)
    return layers_typed

def clone_network(network):
    """Creates a deep copy of the network (cloning the biases and weights)."""
    new_network = create_network()
    # Copy parameters for the hidden and output layers (l2 and l3)
    for layer_idx in [1, 2]:
        original_layer = network[layer_idx]
        new_layer = new_network[layer_idx]
        for i in range(len(original_layer)):
            new_layer[i].bias = original_layer[i].bias
            new_layer[i].weights = original_layer[i].weights.copy()
    return new_network

def crossover(network1, network2):
    """
    Performs crossover between two networks.
    For each neuron in layers 2 and 3, each parameter is chosen from one of the two parents.
    """
    child = create_network()
    for layer_idx in [1, 2]:
        layer1 = network1[layer_idx]
        layer2 = network2[layer_idx]
        child_layer = child[layer_idx]
        for i in range(len(layer1)):
            # For bias: choose randomly from one parent
            if np.random.rand() < 0.5:
                child_layer[i].bias = layer1[i].bias
            else:
                child_layer[i].bias = layer2[i].bias
            # For each weight:
            weights1 = layer1[i].weights
            weights2 = layer2[i].weights
            new_weights = np.empty_like(weights1)
            for j in range(len(weights1)):
                if np.random.rand() < 0.5:
                    new_weights[j] = weights1[j]
                else:
                    new_weights[j] = weights2[j]
            child_layer[i].weights = new_weights
    return child

def mutate(network, mutation_rate=0.1, mutation_strength=0.1):
    """
    Mutates the network’s parameters.
    For each parameter (bias and each weight in layers 2 and 3) with probability mutation_rate,
    adds a small random Gaussian perturbation.
    """
    for layer_idx in [1, 2]:
        layer = network[layer_idx]
        for neuron in layer:
            if np.random.rand() < mutation_rate:
                neuron.bias += np.float32(mutation_strength * np.random.randn())
            for j in range(len(neuron.weights)):
                if np.random.rand() < mutation_rate:
                    neuron.weights[j] += np.float32(mutation_strength * np.random.randn())

def evaluate_network(network, num_episodes=50, epsilon=0.0):
    """
    Evaluates the network by running it for a number of episodes.
    For each episode the network does a forward pass on a random state, selects an action (greedily
    if epsilon=0), steps the environment, and obtains a reward.
    Returns the average reward over the episodes.
    """
    total_reward = 0.0
    l1 = network[0]
    l2 = network[1]
    l3 = network[2]
    for _ in range(num_episodes):
        state = np.random.rand(10).astype(np.float32)
        # Set input layer values
        for i in range(len(l1)):
            l1[i].val = state[i]
        # Forward pass through hidden and output layers
        forward(l1, l2)
        forward(l2, l3)
        # Collect Q-values from the output layer
        q_values = np.empty(len(l3), dtype=np.float32)
        for i in range(len(l3)):
            q_values[i] = l3[i].val
        # Choose action greedily (epsilon=0) or with slight exploration if desired
        action = np.argmax(q_values) if np.random.rand() > epsilon else np.random.randint(0, len(q_values))
        _, reward, _ = env_step(state, action)
        total_reward += reward
    return total_reward / num_episodes

def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Selects one individual from the population using tournament selection.
    Randomly picks tournament_size individuals and returns the one with the highest fitness.
    """
    indices = np.random.choice(len(population), tournament_size, replace=False)
    best_index = indices[0]
    best_fitness = fitnesses[best_index]
    for idx in indices:
        if fitnesses[idx] > best_fitness:
            best_fitness = fitnesses[idx]
            best_index = idx
    return population[best_index]

# -------------------------
# MAIN GA TRAINING LOOP
# -------------------------

# GA parameters
population_size = 50
num_generations = 100
mutation_rate = 0.1
mutation_strength = 0.1
num_episodes_eval = 50  # Number of episodes to estimate fitness

# Initialize the population with randomly created networks
population = [create_network() for _ in range(population_size)]
fitnesses = [evaluate_network(ind, num_episodes=num_episodes_eval) for ind in population]

for generation in trange(num_generations, desc="GA Generations"):
    new_population = []
    # Elitism: carry over the best network unchanged
    best_idx = np.argmax(fitnesses)
    best_individual = clone_network(population[best_idx])
    new_population.append(best_individual)
    
    # Generate the rest of the new population via selection, crossover, and mutation
    while len(new_population) < population_size:
        parent1 = tournament_selection(population, fitnesses, tournament_size=3)
        parent2 = tournament_selection(population, fitnesses, tournament_size=3)
        child = crossover(parent1, parent2)
        mutate(child, mutation_rate, mutation_strength)
        new_population.append(child)
    
    # Replace the old population with the new one and evaluate fitnesses
    population = new_population
    fitnesses = [evaluate_network(ind, num_episodes=num_episodes_eval) for ind in population]
    print(f"Generation {generation}: Best fitness = {np.max(fitnesses):.3f}")

# After GA evolution, retrieve and test the best network.
best_idx = np.argmax(fitnesses)
best_network = population[best_idx]
print("Final evaluation of the best network:")
final_performance = evaluate_network(best_network, num_episodes=100, epsilon=0.0)
print(f"Average reward: {final_performance:.3f}")

# (Optional) You can also use the GA-evolved network in an RL training step.
# For example, to do one Q-learning step using best_network:
# state = np.random.rand(10).astype(np.float32)
# new_state, reward, done, action = q_learning_step(state, best_network, best_network[0], epsilon=0.1, gamma=0.9, learning_rate=0.01)
