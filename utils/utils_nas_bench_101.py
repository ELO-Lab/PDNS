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

def convert_ind_tri(ind):
    """ Convert an individual to upper-triangular binary matrix"""
    res = np.zeros((7, 7), dtype=int)

    k = 0
    for i in range(7):
        for j in range(i + 1, 7):
            res[i][j] = ind[k]
            k += 1

    return res

def convert_ind_ops(ind):
    """ Convert an individual to an operations"""
    ops = [INPUT]
    for i in range(0, 5):
        ops.append(OPS[ind[i]])
    ops.append(OUTPUT)
    return ops

def convert_ind_model_spec(ind):
    """Convert an individual to a model_spec """
    ops = convert_ind_ops(ind[:5])
    tri = convert_ind_tri(ind[5:])
    model_spec = api.ModelSpec(matrix=tri, ops=ops)
    return model_spec

def check_valid(ind):
    tri = convert_ind_tri(ind[5:])
    ops = list(convert_ind_ops(ind[:5]))
    spec = api.ModelSpec(matrix=tri, ops=ops)
    return API_101.is_valid(spec)

def generate_valid_ind():
        # Random giá trị từ 0 -> 2 tương ứng với CONV3X3, CONV1X1, MAXPOOL3X3
        i = 0
        while i < 26:
            ops = np.random.randint(3, size=OP_SPOTS)
            connection = np.random.randint(2, size=EDGE_SPOTS)
            ind = np.concatenate((ops, connection))
            tri = convert_ind_tri(ind[5:])
            ops = list(convert_ind_ops(ind[:5]))
            spec = api.ModelSpec(matrix=tri, ops=ops)
            if API_101.is_valid(spec):
                return ind
            i += 1

        return ind

def initialize_population( num_individuals):
    pop = []
    for i in range(num_individuals):
        pop.append(generate_valid_ind())
    ### DỪNG CODE TẠI ĐÂY ###

    return np.array(pop)

def load_database():
        database = pickle.load(open('data/NAS_Bench_101/data.p', 'rb'))
        zc_database = pickle.load(open('data/NAS_Bench_101/zc_101.p', 'rb'))
        
        max_min_metrics = {
            'max_params': 49979274,
            'min_params': 227274
        }

        pareto_front = pickle.load(open('data/NAS_Bench_101/[POF_TestAcc_Params]_[NAS101].p', 'rb'))
        pareto_front[:, 1] = (pareto_front[:, 1] - max_min_metrics['min_params']) / (max_min_metrics['max_params'] - max_min_metrics['min_params'])
        
        pareto_front_valacc = pickle.load(open('data/NAS_Bench_101/[POF_ValAcc_Params]_[NAS101].p', 'rb'))
        pareto_front_valacc[:, 1] = (pareto_front_valacc[:, 1] - max_min_metrics['min_params']) / (max_min_metrics['max_params'] - max_min_metrics['min_params'])
        
        pareto_fronts = {
            'val_acc': pareto_front_valacc,
            'test_acc': pareto_front
        }
        
        
        return database, zc_database, pareto_fronts, max_min_metrics

class CustomPolynomialMutation(Mutation):
    
    def __init__(self, prob=0.9, eta=20, at_least_once=False, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.at_least_once = at_least_once
        self.eta = Real(eta, bounds=(3.0, 30.0), strict=(1.0, 100.0))

    def _do(self, problem, X, params=None, **kwargs):
        X = X.astype(float)
        Xp = np.copy(X)  # Copy of the original population
        eta = get(self.eta, size=len(X))
        prob_var = self.get_prob_var(problem, size=len(X))

        for i in range(len(X)):  # Loop through each individual
            for _ in range(26):  # Try mutation up to 26 times for each individual
                Xp[i] = mut_pm(X[i].reshape(1, -1), problem.xl, problem.xu, np.array([eta[i]]), np.array([prob_var[i]]), at_least_once=self.at_least_once)
                if check_valid(np.around(Xp[i]).astype(int)):
                    break
            else:  # If the loop completed without a break (i.e., all 26 attempts failed for this individual)
                Xp[i] = X[i]  # Use original individual

        return Xp
    


class CustomTwoPointCrossover(Crossover):
    
    def __init__(self, n_points=2, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.n_points = n_points

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape

        Xp = np.empty_like(X)

        for i in range(n_matings):
            for _ in range(26):  # Try crossover up to 26 times
                # start point of crossover
                r = np.sort(np.random.choice(n_var - 1, self.n_points, replace=False) + 1)
                r = np.append(r, n_var)

                # the mask do to the crossover
                M = np.full(n_var, False)

                # create for each individual the crossover range
                j = 0
                while j < len(r) - 1:
                    a, b = r[j], r[j + 1]
                    M[a:b] = True
                    j += 2

                # Perform the crossover
                Xp[0, i, M] = X[1, i, M]
                Xp[0, i, ~M] = X[0, i, ~M]
                Xp[1, i, M] = X[0, i, M]
                Xp[1, i, ~M] = X[1, i, ~M]


                if check_valid(Xp[0, i]) and check_valid(Xp[1, i]):
                    break
            else:  # If the loop completed without a break (i.e., all 26 attempts failed)
                Xp[:, i, :] = X[:, i, :]  # Use parents as children

        return Xp