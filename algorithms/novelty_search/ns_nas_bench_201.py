import torch
import os
import pickle
import json
from scipy.spatial import distance
import time

from algorithms.base_algo import BaseAlgo
from problems.nas_bench_201 import NASBench201
from algorithms.base_novelty_search import BaseNoveltySearch
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from utils.utils_nas_bench_201 import *
from utils.utils import *

class NoveltySearch201(BaseNoveltySearch):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)    
        self.database, self.zc_database, self.pareto_fronts_testacc, self.pareto_fronts_valacc, self.max_min_metrics = self._load_database()        
        self.log_archs = []
        
    def _load_database(self):
        database, pareto_fronts_valacc = {}, {}
        max_min_metrics = pickle.load(open('data/NAS_Bench_201/max_min_measures.pickle', 'rb'))
        for dataset in self.datasets:
            database[dataset] = pickle.load(open(f'data/NAS_Bench_201/[{dataset}]_database.p', 'rb'))
            pareto_fronts_valacc[dataset] = pickle.load(open(f'data/NAS_Bench_201/[POF_ValAcc_FLOPs]_[NAS201_{dataset}].p', 'rb'))
            pareto_fronts_valacc[dataset][:, 1] = (pareto_fronts_valacc[dataset][:, 1] - max_min_metrics[dataset]['flops_min']) / (max_min_metrics[dataset]['flops_max'] - max_min_metrics[dataset]['flops_min'])
            if dataset != 'ImageNet16-120':
                pareto_fronts_valacc[dataset][:, 0] /= 100
                
        zc_database = json.load(open('data/NAS_Bench_201/zc_nasbench201.json', 'r'))
        
        pareto_fronts_testacc = json.load(open('data/NAS_Bench_201/nb201_pf_norm.json', 'r'))
        
        
        return database, zc_database, pareto_fronts_testacc, pareto_fronts_valacc, max_min_metrics
    
    def _evaluate_and_filter_true_archive(self, var_archive, dataset):
        # Evaluate all architectures in the var archive
        performance, _ = self.evaluate_population(var_archive, metric='test_acc', dataset=dataset, epoch=200)
        complexity, _ = self.evaluate_population(var_archive, metric='flops', dataset=dataset)

        error = 1 - performance
        complexity_norm = (complexity - self.max_min_metrics[dataset]['flops_min']) / (self.max_min_metrics[dataset]['flops_max'] - self.max_min_metrics[dataset]['flops_min'])

        obj_test_archive = list(zip(error, complexity_norm))
        
        non_dominated_var_test_archive, non_dominated_obj_test_archive = remove_dominated_from_archive(var_archive, obj_test_archive)

        return np.array(non_dominated_var_test_archive), np.array(non_dominated_obj_test_archive)
    
    def evaluate_arch(self, arch_var,  metric, dataset, epoch=None):

        arch_str = convert_arch_var_to_str(arch_var)
        if metric in ['train_acc', 'val_acc', 'test_acc', 'flops']:
            arch_var_str = convert_arch_var_to_arch_var_str(arch_var)
            
            if metric in ['flops']:
                score = self.database[dataset]['200'][arch_var_str]['FLOPs']
                op_indices = str(convert_str_to_op_indices(arch_str))
                time = self.zc_database[dataset][op_indices][metric]['time']
            else:
                if metric in ['val_acc'] and epoch not in [200]:
                    score = self.database[dataset]['200'][arch_var_str][metric][epoch]
                else:
                    score = self.database[dataset][str(epoch)][arch_var_str][metric][-1]

                time = self.database[dataset]['200'][arch_var_str]['train_time'] * epoch
                if dataset in ['cifar100', 'ImageNet16-120']:
                    time += self.database[dataset]['200'][arch_var_str]['val_time']
        else:
            op_indices = str(convert_str_to_op_indices(arch_str))
            score = self.zc_database[dataset][op_indices][metric]['score']
            time = self.zc_database[dataset][op_indices][metric]['time']
                           
        return score, time
    
    def evaluate_population(self, var_pop, metric, dataset, epoch=None):
        values = np.array([self.evaluate_arch(var_ind, metric, dataset, epoch) for var_ind in var_pop])
        scores = values[:, 0]
        times = values[:, 1]
        return scores, times
    
    def initialize_population(self):
        var_pop = np.random.randint(5, size=(self.pop_size, self.n_var))
        return var_pop

    def _uniform_crossover(self, pop):
        
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
            
            for idx in range(0, num_parameters):
                r = np.random.rand()
                if r < 0.5:
                    temp = offspring2[idx]
                    offspring2[idx] = offspring1[idx]
                    offspring1[idx] = temp

            offspring.append(offspring1)
            offspring.append(offspring2)



        offspring = np.array(offspring)
        return offspring

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
                    possible_values = list(range(5))
                    possible_values.remove(offspring[i][j])

                    offspring[i][j] = np.random.choice(possible_values)

        mutated_offspring = offspring
        return mutated_offspring
    
    def _calc_performance_indicators(self, elitist_archive, pareto_fronts_testacc):
        
    
        get_igd = IGD(pareto_fronts_testacc)
        get_igd_plus = IGDPlus(pareto_fronts_testacc)
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

            test_var_archive_dict, test_obj_archive_dict, val_var_archive_dict, val_obj_archive_dict, indicators = {}, {}, {}, {}, {}
            for dataset in self.datasets:
                test_var_archive, test_obj_archive = self._evaluate_and_filter_true_archive(var_archive, dataset)
                test_var_archive_dict[dataset] = test_var_archive
                test_obj_archive_dict[dataset] = test_obj_archive
                pareto_front_testacc = np.array(self.pareto_fronts_testacc[dataset])
  
                indicators[dataset] = self._calc_performance_indicators(test_obj_archive, pareto_front_testacc)
            self.data["indicators"].append(indicators)
            self.data["var_archive"].append(var_archive)
            self.data["obj_archive"].append(obj_archive)
            self.data["test_var_archive"].append(test_var_archive_dict)
            self.data["test_obj_archive"].append(test_obj_archive_dict)

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
        var_pop = self.initialize_population()
        
        
        scores_pop = np.zeros((len(var_pop), self.n_objs))
        
        for i, var_ind in enumerate(var_pop):
            for j, metric in enumerate(self.objective_names):
                if metric in ['synflow', 'jacov', 'snip', 'fisher', 'val_acc']:
                    score, performance_time = self.evaluate_arch(var_ind, metric=metric, dataset=self.dataset, epoch=12)
                    score *= -1
                    scores_pop[i][j] = score
                elif metric in ['flops']:
                    score, complexity_time = self.evaluate_arch(var_ind, metric=metric, dataset=self.dataset)
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
                    if metric in ['synflow', 'jacov', 'snip', 'fisher', 'val_acc']:
                        score, performance_time = self.evaluate_arch(var_ind, metric=metric, dataset=self.dataset, epoch=12)
                        score *= -1
                        scores_offspring[i][j] = score
                    elif metric in ['flops']:
                        score, complexity_time = self.evaluate_arch(var_ind, metric=metric, dataset=self.dataset)
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
        
        
        