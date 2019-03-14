#!/usr/bin/env python

import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from skimage import measure
from skimage.future import graph

class Game:
    # Initialize a `FloodIt` game of board `size` n×n with a number `n_colors`
    # of different colors.
    def __init__(self, size, n_colors, seed=None):
        self.size = size
        self.n_colors = n_colors

        board_shape = (size, size)

        if seed is not None:
            np.random.seed(1)

        # We start from 1 because most image segmentation libraries consider a
        # value of 0 as background.
        self.board = np.random.randint(1, n_colors+1, board_shape)

        root_value = 1
        board_root_value = self.board[0,0]

        # Make sure that the value of root cell is always 1
        # for easier consumption of the graph algorithms
        if root_value is not board_root_value:
            all_of_root = self.board == root_value
            all_of_board_root = self.board == board_root_value

            self.board[all_of_root] = board_root_value
            self.board[all_of_board_root] = root_value

    # Return a grid of the same dimensions as self.board, with contiguous areas of
    # the same color carrying a unique label.
    def connected_regions(self):
        return measure.label(self.board, neighbors=4)

# Look at all regions adjacent to the regions with `reference_label`, and sum
# up the areas of those regios that share the same value (color). Return a
# dictionary mapping value -> area.
def __get_area_adjacent_values(board, reference_label, adjacency_graph, region_properties):
    label_value = {}
    label_area = {}
    for prop in region_properties:
        x, y = prop.coords[0]
        label_value[prop.label] = board[x,y]
        label_area[prop.label] = prop.area

    # Regions adjacent to the reference region can have the same value, but not
    # be connected to each other. The greedy algorithm adds the area of all
    # such regions.
    adjacent_value_area = defaultdict(int)

    for _, node in adjacency_graph.edges(reference_label):
        value = label_value[node]
        adjacent_value_area[value] += label_area[node]

    return adjacent_value_area

# A function implementing a greedy strategy.
# Takes a `Game`, and yields the next state of it/its board where it attempts to color the maximum
# possible area adjacent to the root.
def greedy(game):
    connected_regions = game.connected_regions()
    adjacency_graph = graph.rag_mean_color(game.board, connected_regions, connectivity=1)
    props = measure.regionprops(connected_regions)

    root_label = connected_regions[0, 0]

    area_adjacent_values = __get_area_adjacent_values(game.board, root_label, adjacency_graph, props)
    largest_adjacent_value = max(area_adjacent_values.keys(), key = lambda k: area_adjacent_values[k])

    coords_of_root_region = connected_regions == root_label
    game.board[coords_of_root_region] = largest_adjacent_value

    yield game

# Takes an adjacency graph and a label, and looks at all possible paths from
# the node with that label to all other nodes in the graph. Returns the longest
# such path (or the first, if more than one have the same length).
def __find_longest_path(adjacency_graph, root_label):
    # Find the longest path we can walk from the root
    longest_path = None
    for target_node in adjacency_graph.nodes:
        if target_node is not root_label:

            new_path = max(nx.all_shortest_paths(adjacency_graph, root_label, target_node), key=len)

            if longest_path is not None:
                longest_path = max(longest_path, new_path, key=len)
            else:
                longest_path = new_path

    return longest_path

# A function implementing a greedy strategy.
# Takes a `Game`, and yields the next state of it/its board.
# Adjacent areas are translated into an adjacency graph; after obtaining the
# longest path possible path from the root node to any of the other nodes in
# the graph, one step along that path is taken, areas are recolored, nodes
# merged, and a new longest path is constructed.
# possible area adjacent to the root.
def smart(game):
    connected_regions = game.connected_regions()
    properties = measure.regionprops(connected_regions)

    adjacency_graph = graph.rag_mean_color(connected_regions, connected_regions, connectivity=1)

    root_label = connected_regions[0, 0]

    label_to_value = {}
    for prop in properties:
        x, y = prop.coords[0]
        label_to_value[prop.label] = game.board[x,y]

    while len(adjacency_graph.nodes) > 1:
        longest_path = __find_longest_path(adjacency_graph, root_label)
        # This target label is guaranteed to exist if there is more than one node in the graph and
        # the graph is fully connected.
        target_label = longest_path[1]
        target_value = label_to_value[target_label]

        # Merge the root_label into the target_label
        # The target_label becomes the new root label
        adjacency_graph = nx.algorithms.minors.contracted_nodes(adjacency_graph, target_label, root_label)

        coords_of_root_region = connected_regions == root_label
        connected_regions[coords_of_root_region] = target_label
        game.board[coords_of_root_region] = target_value
        root_label = target_label

        # Iterate over the neighbors of the new contracted root node and merge them if they have the same value as root
        for neighbor in adjacency_graph[target_label].keys():
            if label_to_value[neighbor] == target_value:
                adjacency_graph = nx.algorithms.minors.contracted_nodes(adjacency_graph, root_label, neighbor)
                connected_regions[connected_regions == neighbor] = root_label

        yield game

def run_strategy(game, strategy):
    while not np.all(game.board == game.board[0,0]):
        for g in strategy(game):
            yield g
        game = g

strategies = {
    'greedy': greedy,
    'smart': smart,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate playing the Flood-It game. Click to advance')
    parser.add_argument('-n', '--size', type=int, required=True, help='The number n of tiles along a side of the n×n grid.')
    parser.add_argument('-c', '--colors', type=int, required=True, help='The number of colors in the game')
    parser.add_argument('-s', '--strategy', choices=['greedy', 'smart'], default='smart', help='The strategy to be used; default: %(default)s')
    parser.add_argument('-f', '--fixed', action='store_const', const=1, help='Used a fixed seed. This way, the board will always look the same.')

    args = parser.parse_args()

    game = Game(args.size, args.colors, args.fixed)
    strategy = strategies[args.strategy]

    img = plt.imshow(game.board)
    plt.waitforbuttonpress()
    for game in run_strategy(game, strategy):
        img.set_data(game.board)
        plt.draw()
        plt.waitforbuttonpress()
