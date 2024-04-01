import torch
import os
import pickle
import json
from scipy.spatial import distance
import time

from algorithms.base_algo import BaseAlgo
from problems.nas_bench_101 import NASBench101
from algorithms.novelty_search.ns_nas_bench_101 import NoveltySearch101
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from utils.utils_nas_bench_1shot1 import *
from utils.utils import *

class NoveltySearch1Shot1(NoveltySearch101):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)    
        if self.search_space==1:
            self.CONFIG_SPACE = config_space_1
        elif self.search_space==2:
            self.CONFIG_SPACE = config_space_2
        elif self.search_space==3:
            self.CONFIG_SPACE = config_space_3
        
        
        self.xl = np.array([0] * self.n_var)
        
        if self.search_space==3:
            self.xu = np.array([2] * 5 + [1] + [2] + [5] + [9] + [14])
        elif self.search_space==2:
            self.xu = np.array([2] * 5 + [1] + [2] + [5] + [9])
        elif self.search_space==1:
            self.xu = np.array([2] * 5 + [2] + [5] + [9])
            
        self.database, self.zc_database, self.pareto_fronts = load_database(self.search_space)
        
    def create_adjacency_matrix(self, ind):
        if self.search_space==3:
            parents = { '0': [], '1': [0]}
        elif self.search_space==2:
            parents = { '0': [], '1': [0]}
        elif self.search_space==1:
            parents = {'0': [], '1': [0], '2': [0, 1]}
        
        for node, choice in enumerate(range(5, len(ind)), len(parents)):
            parents[f'{node}'] = self.CONFIG_SPACE[f'{node}'][ind[choice]]
            
        matrix = create_nasbench_adjacency_matrix_with_loose_ends(parents)
        return matrix
    
    def create_spec(self, arch_var):
        ops = create_operations_list(arch_var)
        adjency_matrix = self.create_adjacency_matrix(arch_var)
        if self.search_space in [1, 2]:
            adjency_matrix = upscale_to_nasbench_format(adjency_matrix)
        spec = api.ModelSpec(matrix=adjency_matrix, ops=ops)
        return spec 

    def evaluate_arch(self, arch_var, metric, epoch=None):
        spec = self.create_spec(arch_var)
        hash_key = API_101.get_module_hash(spec)

        if metric in ['train_acc', 'val_acc', 'test_acc']:
            score = self.database[str(epoch)][hash_key][metric]
            time = self.database[str(epoch)][hash_key]['train_time']
        elif metric in ['n_params']:
            score = self.database['108'][hash_key][metric]
            time = self.estimate_times[metric]
        else:
            if metric in ['synflow', 'snip', 'fisher', 'grad_norm']:
                score = np.log10(self.zc_database[hash_key][metric])
            else:
                score = self.zc_database[hash_key][metric]
                if score != -1e8:
                    score = -1  * np.log10(-1 * score)                    
            time = self.estimate_times[metric]

        return score, time
    
    def _evaluate_and_filter_test_archive(self, var_archive):
        testacc, _ = self.evaluate_population(var_archive, metric='test_acc', epoch=108)
        params, _ = self.evaluate_population(var_archive, metric='n_params')
        
        testerr = 1 - testacc
        params_norm = params / 1e8

        obj_test_archive = list(zip(testerr, params_norm))

        non_dominated_var_test_archive, non_dominated_obj_test_archive = remove_dominated_from_archive(var_archive, obj_test_archive)

        return np.array(non_dominated_var_test_archive), np.array(non_dominated_obj_test_archive)

    def _two_point_crossover(self, pop):
        num_individuals = len(pop)
        num_parameters = len(pop[0])
        indices = np.arange(num_individuals)
        np.random.shuffle(indices)
        offspring = []

        for i in range(0, num_individuals, 2):
            idx1 = indices[i]
            idx2 = indices[i+1]
            offspring1 = list(pop[idx1])
            offspring2 = list(pop[idx2])

            if num_parameters > 1:
                point1, point2 = sorted(np.random.choice(range(num_parameters), 2, replace=False))

                for idx in range(point1, point2):
                    temp = offspring2[idx]
                    offspring2[idx] = offspring1[idx]
                    offspring1[idx] = temp

            offspring.append(offspring1)
            offspring.append(offspring2)

        offspring = np.array(offspring)
        return offspring

    def _mutation(self, offspring, mutation_rate):
        num_individuals = offspring.shape[0]
        num_parameters = offspring.shape[1]

        for i in range(num_individuals):
            for j in range(num_parameters):
                if np.random.rand() < mutation_rate:
                    possible_values = list(range(self.xu[j] + 1))
                    possible_values.remove(offspring[i][j])

                    offspring[i][j] = np.random.choice(possible_values)

        mutated_offspring = offspring
        return mutated_offspring
    
    def _initialize_population(self, pop_size):
        random_array = np.zeros((pop_size, self.n_var), dtype=int)

        for i in range(self.n_var):
            random_array[:, i] = np.random.randint(self.xl[i], self.xu[i] + 1, pop_size)

        return random_array

    def run(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        start_time = time.time()
        cum_time = 0
        self.data['time'].append(start_time)
        var_pop = self._initialize_population(self.pop_size)
        
        
        scores_pop = np.zeros((len(var_pop), self.n_objs))
        
        for i, var_ind in enumerate(var_pop):
            for j, metric in enumerate(self.objective_names):
                if metric in ['synflow', 'jacob_cov', 'snip', 'fisher', 'val_acc']:
                    score, performance_time = self.evaluate_arch(var_ind, metric=metric, epoch=12)
                    score *= -1
                    scores_pop[i][j] = score
                elif metric in ['n_params']:
                    score, complexity_time = self.evaluate_arch(var_ind, metric=metric)
                    scores_pop[i][j] = score
            algo_time = time.time()
            cum_time += performance_time + complexity_time
            
            self.data['time'].append(algo_time + cum_time)
                
        self._archive_check(var_pop, scores_pop)
        
        
        for i in range(self.n_gen):
            var_offspring = self._two_point_crossover(var_pop)
            var_offspring = self._mutation(var_offspring, 1/6)

            scores_offspring = np.zeros((len(var_offspring), self.n_objs))
        
            for i, var_ind in enumerate(var_offspring):
                for j, metric in enumerate(self.objective_names):
                    if metric in ['synflow', 'jacob_cov', 'snip', 'fisher', 'val_acc']:
                        score, performance_time = self.evaluate_arch(var_ind, metric=metric, epoch=12)
                        score *= -1
                        scores_offspring[i][j] = score
                    elif metric in ['n_params']:
                        score, complexity_time = self.evaluate_arch(var_ind, metric=metric)
                        scores_offspring[i][j] = score
                algo_time = time.time()
                if len(self.log_archs)==0  or var_ind.tolist() not in self.log_archs:
                    cum_time += performance_time + complexity_time
                    self.log_archs.append(var_ind.tolist())        
                self.data['time'].append(algo_time + cum_time)
                                    
            self._archive_check(var_offspring, scores_offspring)


            pool = np.vstack((var_pop, var_offspring))
            scores_pool = np.vstack((scores_pop, scores_offspring))

            pool_fitness = np.array(self._average_archive_distances(scores_pool, self.data["obj_archive"][-1], pool, self.data["var_archive"][-1]))

            pool_indices = self._tournament_selection(pool, pool_fitness, self.selection_size, self.tournament_size)

      
            var_pop = pool[pool_indices, :]
            pop_fitness = pool_fitness[pool_indices]
            scores_pop = scores_pool[pool_indices, :]

        algo_time = time.time()
        self.data['time'].append(algo_time + cum_time)
                        
        if self.save_result:
            if self.problem_name in ['NASBench201', 'TransNASBench101']:
                save_dir = f'results/{self.problem_name}/{self.method}/{self.performance_criteria}/{self.dataset}/'
            elif  self.problem_name in ['NASBench1Shot1']:
                save_dir = f'results/{self.problem_name}/{self.method}/{self.performance_criteria}/{self.search_space}/'
            else:
                save_dir = f'results/{self.problem_name}/{self.method}/{self.performance_criteria}'
            
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f'seed_{self.seed}.pkl'), 'wb') as f:
                pickle.dump(self.data, f)
        
        
    