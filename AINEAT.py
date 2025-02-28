import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, Tuple

# -------------------------------
# Environment Definition
# -------------------------------

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

# -------------------------------
# NEAT Genome Definitions
# -------------------------------

@dataclass
class NodeGene:
    id: int
    type: str  # 'input', 'hidden', or 'output'
    activation: float = 0.0

@dataclass
class ConnectionGene:
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True
    innovation: int = 0

@dataclass
class Genome:
    nodes: Dict[int, NodeGene] = field(default_factory=dict)
    connections: Dict[Tuple[int, int], ConnectionGene] = field(default_factory=dict)

# Global innovation counter
global_innovation = 0
def get_new_innovation():
    global global_innovation
    global_innovation += 1
    return global_innovation

# -------------------------------
# Genome Creation and Mutation
# -------------------------------

def create_initial_genome(num_inputs=10, num_outputs=3) -> Genome:
    """
    Creates an initial genome with all input nodes connected to output nodes.
    """
    genome = Genome()
    # Create input nodes
    for i in range(num_inputs):
        genome.nodes[i] = NodeGene(id=i, type='input')
    # Create output nodes (IDs continue)
    for j in range(num_inputs, num_inputs + num_outputs):
        genome.nodes[j] = NodeGene(id=j, type='output')
    # Fully connect inputs to outputs
    for i in range(num_inputs):
        for j in range(num_inputs, num_inputs + num_outputs):
            innov = get_new_innovation()
            genome.connections[(i, j)] = ConnectionGene(
                in_node=i,
                out_node=j,
                weight=np.random.randn(),
                enabled=True,
                innovation=innov
            )
    return genome

def mutate_weights(genome: Genome, mutation_rate=0.8, mutation_strength=0.5):
    """
    Perturb the weight of each connection with some probability.
    """
    for conn in genome.connections.values():
        if random.random() < mutation_rate:
            conn.weight += np.random.randn() * mutation_strength

def mutate_add_connection(genome: Genome, max_tries=100):
    """
    Attempts to add a new connection between two nodes that is not already present.
    """
    node_ids = list(genome.nodes.keys())
    for _ in range(max_tries):
        in_node = random.choice(node_ids)
        out_node = random.choice(node_ids)
        # Prevent invalid connections: output-to-input, self–connections
        if genome.nodes[in_node].type == 'output' or genome.nodes[out_node].type == 'input' or in_node == out_node:
            continue
        if (in_node, out_node) in genome.connections:
            continue
        innov = get_new_innovation()
        genome.connections[(in_node, out_node)] = ConnectionGene(
            in_node=in_node,
            out_node=out_node,
            weight=np.random.randn(),
            enabled=True,
            innovation=innov
        )
        return  # Successful mutation
    # No valid connection found; do nothing.

def mutate_add_node(genome: Genome):
    """
    Splits an existing connection by disabling it and adding a new node between.
    """
    if not genome.connections:
        return
    connection = random.choice(list(genome.connections.values()))
    if not connection.enabled:
        return
    connection.enabled = False  # Disable the old connection
    new_node_id = max(genome.nodes.keys()) + 1
    genome.nodes[new_node_id] = NodeGene(id=new_node_id, type='hidden')
    # Add connection from the original in_node to the new node
    innov1 = get_new_innovation()
    genome.connections[(connection.in_node, new_node_id)] = ConnectionGene(
        in_node=connection.in_node,
        out_node=new_node_id,
        weight=1.0,  # Often initialized to 1.0
        enabled=True,
        innovation=innov1
    )
    # Add connection from the new node to the original out_node
    innov2 = get_new_innovation()
    genome.connections[(new_node_id, connection.out_node)] = ConnectionGene(
        in_node=new_node_id,
        out_node=connection.out_node,
        weight=connection.weight,  # Inherit the old weight
        enabled=True,
        innovation=innov2
    )

def crossover(genome1: Genome, genome2: Genome) -> Genome:
    """
    Performs a simple crossover between two genomes.
    For each connection gene in genome1, if a matching gene exists in genome2,
    it is chosen at random; otherwise, the gene from genome1 is used.
    (Assumes genome1 is at least as fit as genome2.)
    """
    child = Genome()
    # Inherit nodes from genome1 (assumes both genomes share the same input/output nodes)
    for node_id, node in genome1.nodes.items():
        child.nodes[node_id] = NodeGene(id=node.id, type=node.type)
    # Inherit connections
    for key, conn in genome1.connections.items():
        if key in genome2.connections and random.random() < 0.5:
            inherited = genome2.connections[key]
        else:
            inherited = conn
        child.connections[key] = ConnectionGene(
            in_node=inherited.in_node,
            out_node=inherited.out_node,
            weight=inherited.weight,
            enabled=inherited.enabled,
            innovation=inherited.innovation
        )
    return child

# -------------------------------
# Forward Pass (Network Execution)
# -------------------------------

def forward_pass(genome: Genome, input_values: np.ndarray) -> np.ndarray:
    """
    Executes a forward pass through the network represented by the genome.
    Nodes are processed in order of their IDs (assumed topologically sorted).
    Applies ReLU activation for hidden and output nodes.
    """
    # Set activations for input nodes
    for node in genome.nodes.values():
        if node.type == 'input':
            # Assumes input_values are ordered by node id
            node.activation = input_values[node.id]
    # Process other nodes (assumes nodes are roughly sorted by id)
    sorted_nodes = sorted(genome.nodes.values(), key=lambda n: n.id)
    for node in sorted_nodes:
        if node.type == 'input':
            continue
        total = 0.0
        # Sum contributions from all enabled incoming connections
        for conn in genome.connections.values():
            if conn.enabled and conn.out_node == node.id:
                total += genome.nodes[conn.in_node].activation * conn.weight
        node.activation = max(0.0, total)  # ReLU activation
    # Collect outputs (nodes with type 'output')
    outputs = [node.activation for node in sorted_nodes if node.type == 'output']
    return np.array(outputs, dtype=np.float32)

# -------------------------------
# Genome Evaluation
# -------------------------------

def evaluate_genome(genome: Genome, num_episodes=50, epsilon=0.0) -> float:
    """
    Evaluates the genome by running it in the environment for a number of episodes.
    The network outputs (Q-values) are used to select an action (greedy or epsilon–greedy).
    Returns the average reward over episodes.
    """
    total_reward = 0.0
    for _ in range(num_episodes):
        state = np.random.rand(10).astype(np.float32)
        q_values = forward_pass(genome, state)
        action = np.argmax(q_values) if np.random.rand() > epsilon else np.random.randint(0, len(q_values))
        _, reward, _ = env_step(state, action)
        total_reward += reward
    return total_reward / num_episodes

# -------------------------------
# Main Evolutionary Loop
# -------------------------------

def main():
    population_size = 50
    num_generations = 1000

    # Initialize the population with random genomes
    population = [create_initial_genome() for _ in range(population_size)]
    fitnesses = [evaluate_genome(genome, num_episodes=50) for genome in population]

    for generation in range(num_generations):
        new_population = []
        # Elitism: carry over the best genome
        best_idx = np.argmax(fitnesses)
        best_genome = population[best_idx]
        new_population.append(best_genome)
        
        # Generate the rest of the new population via crossover and mutation
        while len(new_population) < population_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2)
            mutate_weights(child)
            if random.random() < 0.05:
                mutate_add_connection(child)
            if random.random() < 0.03:
                mutate_add_node(child)
            new_population.append(child)
        
        population = new_population
        fitnesses = [evaluate_genome(genome, num_episodes=50) for genome in population]
        print(f"Generation {generation}: Best fitness = {np.max(fitnesses):.3f}")

    print("Final evaluation:")
    best_genome = population[np.argmax(fitnesses)]
    final_performance = evaluate_genome(best_genome, num_episodes=100)
    print(f"Average reward: {final_performance:.3f}")

if __name__ == '__main__':
    main()
