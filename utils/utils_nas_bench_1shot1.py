import numpy as np
import pickle

from pymoo.core.mutation import Mutation
from pymoo.core.variable import get, Real
from pymoo.operators.mutation.pm import mut_pm
from pymoo.core.crossover import Crossover
from pymoo.util.misc import crossover_mask

from data.NAS_Bench_101.nasbench import wrap_api as api


# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
EDGE_SPOTS = int(NUM_VERTICES * (NUM_VERTICES - 1) / 2)   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
MIN_PARAMS, MAX_PARAMS = 227274, 49979274

OPS = {0: CONV3X3, 1: CONV1X1, 2: MAXPOOL3X3}

API_101 = api.NASBench_()


def create_operations_list(individual):
    operations = [INPUT] + [OPS[op] for op in individual[:5]] + [OUTPUT]
    return operations

config_space_3 = {
    '2': [(0,), (1,)],
    '3': [(0,), (1,), (2,)],
    '4': [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    '5': [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
    '6': [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
}

config_space_2 = {
    '2': [(0,), (1,)],
    '3': [(0, 1), (0, 2), (1, 2)],
    '4': [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    '5': [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]
}

config_space_1 = {
    '3': [(0, 1), (0, 2), (1, 2)],
    '4': [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    '5': [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
}

def create_nasbench_adjacency_matrix_with_loose_ends(parents):
    adjacency_matrix = np.zeros([len(parents), len(parents)], dtype=int)
    for node, node_parents in parents.items():
        for parent in node_parents:
            adjacency_matrix[parent, int(node)] = 1
    
    return adjacency_matrix

def upscale_to_nasbench_format(adjacency_matrix):
    """
    The search space uses only 4 intermediate nodes, rather than 5 as used in nasbench
    This method adds a dummy node to the graph which is never used to be compatible with nasbench.
    :param adjacency_matrix:
    :return:
    """
    return np.insert(
        np.insert(adjacency_matrix, 5, [0, 0, 0, 0, 0, 0], axis=1),
        5, [0, 0, 0, 0, 0, 0, 0], axis=0)
    


def load_database(search_space):
        database = pickle.load(open('data/NAS_Bench_101/data.p', 'rb'))
        zc_database = pickle.load(open('data/NAS_Bench_101/zc_101.p', 'rb'))

        pareto_front = pickle.load(open(f'data/NAS_Bench_1Shot1/pareto_front(testing)_NAS101-{search_space}.p', 'rb'))
        
        pareto_fronts = {
            'test_acc': pareto_front
        }
        
        
        return database, zc_database, pareto_fronts