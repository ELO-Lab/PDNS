import torch
import os
import pickle


from algorithms.base_algo import BaseAlgo
from problems.nas_bench_201 import NASBench201

from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from utils.utils_nas_bench_201 import *
from utils.utils import *

class BaseNoveltySearch(BaseAlgo):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)    
        self.var_pop = None
        self.datasets = ['cifar10', 'cifar100', 'ImageNet16-120'] if self.dataset=='cifar10' else [self.dataset]
        self.data = {"var_pop": [], "obj_pop": [], "var_archive": [], "obj_archive": [], "test_var_archive": [], "test_obj_archive": [], "indicators": [], "time": []}
        self.selection_size = self.pop_size
        self.tournament_size = 2
        self.ref_point = [1.05, 1.05]
        self.time = []
        
    
    
    def _initialize_population(self):
        raise NotImplementedError

    def _better_fitness(self, fitness_1, fitness_2, maximization=True):
        if maximization:
            if fitness_1 > fitness_2:
                return True
        else:
            if fitness_1 < fitness_2:
                return True

        return False

    def _tournament_selection(self, pop, pop_fitness, selection_size, tournament_size):
            num_individuals = len(pop)
            indices = np.arange(num_individuals)
            selected_indices = []

            while len(selected_indices) < selection_size:
                np.random.shuffle(indices)

                for i in range(0, num_individuals, tournament_size):
                    best_idx = i
                    for idx in range(1, tournament_size):
                        if self._better_fitness(pop_fitness[indices[i + idx]], pop_fitness[indices[best_idx]]):
                            best_idx = i + idx
                    selected_indices.append(indices[best_idx])

            selected_indices = np.array(selected_indices)


            return selected_indices

    def _uniform_crossover(self, pop):
        raise NotImplementedError

    def _two_point_crossover(self, pop):
        raise NotImplementedError
    
    def _mutation(self, offspring, mutation_rate):
        raise NotImplementedError

    def _converge_pop_fitness(pop_fitness):
        if np.all(pop_fitness == pop_fitness[0]):
            return True
        return False


    def _check_convergence(population):
        return (population == population[0]).all()
