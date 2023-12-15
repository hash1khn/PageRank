import sys
import os
import time
import argparse
import random
from collections import defaultdict
from progress import Progress  # Import the Progress class

def load_graph(args):
    """Load graph from text file

    Parameters:
    args -- arguments named tuple

    Returns:
    A dict mapping a URL (str) to a list of target URLs (str).
    """
    graph = defaultdict(list)
    for line in args.datafile:
        node, target = line.split()
        graph[node].append(target)
    return graph

def print_stats(graph):
    """Print the number of nodes and edges in the given graph"""
    num_nodes = len(graph)
    num_edges = sum(len(targets) for targets in graph.values())
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")

def stochastic_page_rank(graph, args):
    """Stochastic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    args -- arguments named tuple

    Returns:
    A dict that assigns each page its hit frequency

    This function estimates the Page Rank by counting how frequently
    a random walk that starts on a random node will, after n_steps,
    end on each node of the given graph.
    """
    hits = {node: 0 for node in graph}
    progress_bar = Progress(args.repeats, "Calculating Page Ranks")

    for _ in range(args.repeats):
        node = random.choice(list(graph.keys()))
        for _ in range(args.steps):
            node = random_walk(graph, node)
            hits[node] += 1
        progress_bar += 1
        progress_bar.show()

    progress_bar.finish()
    total_hits = sum(hits.values())
    return {node: hit / total_hits for node, hit in hits.items()}

def distribution_page_rank(graph, args):
    """Probabilistic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    args -- arguments named tuple

    Returns:
    A dict that assigns each page its probability to be reached

    This function estimates the Page Rank by iteratively calculating
    the probability that a random walker is currently on any node.
    """
    num_nodes = len(graph)
    initial_prob = 1 / num_nodes
    probabilities = {node: initial_prob for node in graph}

    progress_bar = Progress(args.steps, "Calculating Page Ranks")

    for _ in range(args.steps):
        new_probabilities = {node: 0 for node in graph}
        for node, targets in graph.items():
            if not targets:
                new_probabilities[node] += probabilities[node]
            else:
                probability_to_distribute = probabilities[node] / len(targets)
                for target in targets:
                    new_probabilities[target] += probability_to_distribute
        probabilities = new_probabilities
        progress_bar += 1
        progress_bar.show()

    progress_bar.finish()
    return probabilities

def random_walk(graph, current_node):
    """Perform a random walk from the current node"""
    targets = graph.get(current_node, [])
    return random.choice(targets) if targets else current_node

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimates page ranks from link information")
    parser.add_argument('datafile', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help="Textfile of links among web pages as URL tuples")
    parser.add_argument('-m', '--method', choices=('stochastic', 'distribution'), default='stochastic',
                        help="selected page rank algorithm")
    parser.add_argument('-r', '--repeats', type=int, default=1_000_000, help="number of repetitions")
    parser.add_argument('-s', '--steps', type=int, default=100, help="number of steps a walker takes")
    parser.add_argument('-n', '--number', type=int, default=20, help="number of results shown")

    args = parser.parse_args()
    algorithm = distribution_page_rank if args.method == 'distribution' else stochastic_page_rank

    graph = load_graph(args)

    print_stats(graph)

    start = time.time()
    ranking = algorithm(graph, args)
    stop = time.time()
    elapsed_time = stop - start 

    top = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    sys.stderr.write(f"Top {args.number} pages:\n")
    print('\n'.join(f'{100 * v:.2f}\t{k}' for k, v in top[:args.number]))
    sys.stderr.write(f"Calculation took {elapsed_time:.2f} seconds.\n")
