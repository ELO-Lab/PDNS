import torch
import os
import pickle
import json
from scipy.spatial import distance
import time

from algorithms.base_algo import BaseAlgo
from problems.nas_bench_101 import NASBench101
from algorithms.base_novelty_search import BaseNoveltySearch
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from utils.utils_nas_bench_101 import *
from utils.utils import *

class NoveltySearch101(BaseNoveltySearch):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)    
        self.database, self.zc_database, self.pareto_fronts, self.max_min_metrics = load_database()      
        self.log_archs = []
        self.estimate_times = {
            'n_params': 0.30215823150349097,
            'synflow': 1.4356617034300945,
            'jacob_cov': 2.5207841626097856,
            'snip': 2.028758352457235
        }

    def evaluate_arch(self, arch_var, metric, epoch=None):
        tri = convert_ind_tri(arch_var[5:])
        ops = list(convert_ind_ops(arch_var[:5]))
        spec = api.ModelSpec(matrix=tri, ops=ops)

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
    
    def evaluate_population(self, pop, metric=None, epoch=None):
        values = np.array([self.evaluate_arch(ind, metric, epoch) for ind in pop])
        scores = values[:, 0]
        times = values[:, 1]

        return scores, times
    
    def _evaluate_and_filter_test_archive(self, var_archive):
        testacc, _ = self.evaluate_population(var_archive, metric='test_acc', epoch=108)
        params, _ = self.evaluate_population(var_archive, metric='n_params')
        
        testerr = 1 - testacc
        params_norm = (params - self.max_min_metrics['min_params']) / (self.max_min_metrics['max_params'] - self.max_min_metrics['min_params'])

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
            
            cnt_cross = 0
            while (cnt_cross < 26):
                if num_parameters > 1:
                    point1, point2 = sorted(np.random.choice(range(num_parameters), 2, replace=False))

                    for idx in range(point1, point2):
                        temp = offspring2[idx]
                        offspring2[idx] = offspring1[idx]
                        offspring1[idx] = temp
                    if check_valid(offspring1) and check_valid(offspring2):
                        break
                    cnt_cross += 1

            offspring.append(offspring1)
            offspring.append(offspring2)

        offspring = np.array(offspring)
        return offspring

    def _mutation(self, offspring, mutation_rate):
        num_individuals = offspring.shape[0]
        num_parameters = offspring.shape[1]

        for i in range(num_individuals):
            cnt_mut = 0
            while cnt_mut < 26:
                ind = offspring[i].copy()
                for j in range(num_parameters):
                    if np.random.rand() < mutation_rate:
                        if j < 5:
                            possible_values = list(range(3))
                        else:
                            possible_values = list(range(2))
                        possible_values.remove(offspring[i][j])

                        ind[j] = np.random.choice(possible_values)
                if check_valid(ind):
                    offspring[i] = ind.copy()
                    break
                cnt_mut += 1


        mutated_offspring = offspring
        return mutated_offspring
    def _calc_performance_indicators(self, elitist_archive, pareto_fronts):
        get_igd = IGD(pareto_fronts['test_acc'])
        get_igd_plus = IGDPlus(pareto_fronts['test_acc'])
        get_hv = HV(ref_point=self.ref_point)
    

        indicators = {}
        indicators['igd'] = get_igd(elitist_archive)
        indicators['igd_plus'] = get_igd_plus(elitist_archive)
        indicators['hv'] = get_hv(elitist_archive)

        return indicators
    
    def _archive_check(self, var_pop, obj_pop):
        var_archive = self.data["var_archive"][-1].copy() if self.data["var_archive"] else []
        obj_archive = self.data["obj_archive"][-1].copy() if self.data["obj_archive"] else []
        for var_ind, obj_ind in zip(var_pop, obj_pop):
            var_archive, obj_archive = update_elitist_archive(var_archive, obj_archive, var_ind, obj_ind)
            test_var_archive, test_obj_archive = self._evaluate_and_filter_test_archive(var_archive)
            indicators = self._calc_performance_indicators(test_obj_archive, self.pareto_fronts)
            
            self.data["indicators"].append(indicators)
            self.data["var_archive"].append(var_archive)
            self.data["obj_archive"].append(obj_archive)
            self.data["test_var_archive"].append(test_var_archive)
            self.data["test_obj_archive"].append(test_obj_archive)

        self.data["var_pop"].append(var_pop)
        self.data["obj_pop"].append(obj_pop)
        self.data["log_archs"] = self.log_archs
        
    def _mean_distance_to_front(self, obj_archive, obj_new_point, var_archive, var_new_point):        
        mean_distance = np.mean([distance.euclidean(obj_new_point, obj_point) for obj_point in obj_archive])
        var_new_point = np.array(var_new_point)
        if not array_in_arrays(var_new_point, var_archive):
            mean_distance *= -1

        return mean_distance
    
    def _archive_dist(self, norm_current_ind, norm_log_archive, var_current_ind, var_log_archive):
        dist = self._mean_distance_to_front(norm_log_archive, norm_current_ind, var_log_archive, var_current_ind)
        return dist
    
    def _average_archive_distances(self, current_pop, log_archive, var_current_pop, var_log_archive):
        current_pop = np.array(current_pop)
        log_archive = np.array(log_archive)

        for i in range(self.n_objs - 1):
            current_pop[:, i] = -1 * current_pop[:, i]
            log_archive[:, i] = -1 * log_archive[:, i]
        
        all_values = np.vstack((current_pop, log_archive))
        
        masked_arr = np.ma.masked_equal(all_values, -1e8)

        max_values = np.max(all_values, axis=0)
        min_values = masked_arr.min(axis=0)
        mask = current_pop == -1e8
        
        norm_current_pop = (current_pop - min_values) / (max_values - min_values)
        norm_current_pop[mask] = 0

        mask = log_archive == -1e8
        norm_log_archive = (log_archive - min_values) / (max_values - min_values)
        norm_log_archive[mask] = 0

        for i in range(self.n_objs - 1):
            norm_current_pop[:, i] = 1 - norm_current_pop[:, i]
            norm_log_archive[:, i] = 1 - norm_log_archive[:, i]


        mean_distances = []
        for i in range(len(norm_current_pop)):
            mean_distance = self._archive_dist(norm_current_pop[i], norm_log_archive, var_current_pop[i], var_log_archive)
            mean_distances.append(mean_distance)
        return mean_distances
    
    def run(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        start_time = time.time()
        cum_time = 0
        self.data['time'].append(start_time)
        var_pop = initialize_population(self.pop_size)
        
        
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
            if self.problem_name=='NASBench201':
                save_dir = f'results/{self.problem_name}/{self.method}/{self.performance_criteria}/{self.dataset}/'
            else:
                save_dir = f'results/{self.problem_name}/{self.method}/{self.performance_criteria}'
            
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f'seed_{self.seed}.pkl'), 'wb') as f:
                pickle.dump(self.data, f)
        
        
    