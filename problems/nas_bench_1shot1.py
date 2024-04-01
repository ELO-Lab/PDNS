import pickle
import numpy as np 
import time 

from problems.nas_bench_101 import NASBench101
from utils.utils import *
from utils.utils_nas_bench_1shot1 import *

from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

class NASBench1Shot1(NASBench101):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.xl = np.array([0] * self.n_var)
        
        if self.search_space==1:
            self.CONFIG_SPACE = config_space_1
        elif self.search_space==2:
            self.CONFIG_SPACE = config_space_2
        elif self.search_space==3:
            self.CONFIG_SPACE = config_space_3
            
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
            score = self.zc_database[hash_key][metric]
            time = self.estimate_times[metric]
    
        return score, time
    

    def _evaluate_and_filter_true_archive(self, var_archive):
        testacc, _ = self.evaluate_population(var_archive, metric='test_acc', epoch=108)
        params, _ = self.evaluate_population(var_archive, metric='n_params')
        
        testerr = 1 - testacc
        params_norm = params / 1e8

        obj_test_archive = list(zip(testerr, params_norm))

        non_dominated_var_test_archive, non_dominated_obj_test_archive = remove_dominated_from_archive(var_archive, obj_test_archive)

        return np.array(non_dominated_var_test_archive), np.array(non_dominated_obj_test_archive)
    
    def _evaluate(self, var_pop, out, *args, **kwargs):
    
        all_scores = np.zeros((len(var_pop), self.n_objs))
        for i, var_ind in enumerate(var_pop):
            for j, metric in enumerate(self.objective_names):
                if metric in ['synflow', 'jacob_cov', 'snip', 'fisher', 'val_acc']:
                    score, performance_time = self.evaluate_arch(var_ind, metric=metric, epoch=12)
                    score *= -1
                    all_scores[i][j] = score
                elif metric in ['n_params']:
                    score, complexity_time = self.evaluate_arch(var_ind, metric=metric)
                    all_scores[i][j] = score
            algo_time = time.time()
            
            if len(self.log_archs)==0  or var_ind.tolist() not in self.log_archs:
                    self.cum_time += performance_time + complexity_time
                    self.log_archs.append(var_ind.tolist())       
            self.time.append(algo_time + self.cum_time)
                    

        self._n_gen += 1
    
        out["F"] = all_scores
