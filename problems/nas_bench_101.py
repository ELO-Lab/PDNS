import pickle
import numpy as np 
import time 

from problems.base_nas import BaseNAS
from utils.utils import *
from utils.utils_nas_bench_101 import *

from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

class NASBench101(BaseNAS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.database, self.zc_database, self.pareto_fronts, self.max_min_metrics = load_database()
        self.xl = np.array([0] * 26)
        self.xu = np.array([2] * 5 + [1] * 21)
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
            score = self.zc_database[hash_key][metric]
            time = self.estimate_times[metric]
    
        return score, time
    
    def evaluate_population(self, pop, metric=None, epoch=None):
        values = np.array([self.evaluate_arch(ind, metric, epoch) for ind in pop])
        scores = values[:, 0]
        times = values[:, 1]

        return scores, times
    
    def _calc_performance_indicators(self, elitist_archive, pareto_fronts):
        get_igd = IGD(pareto_fronts['test_acc'])
        get_igd_plus = IGDPlus(pareto_fronts['test_acc'])
        get_hv = HV(ref_point=self.ref_point)
    

        indicators = {}
        indicators['igd'] = get_igd(elitist_archive)
        indicators['igd_plus'] = get_igd_plus(elitist_archive)
        indicators['hv'] = get_hv(elitist_archive)
    
    
        return indicators
    
    def _evaluate_and_filter_true_archive(self, var_archive):
        testacc, _ = self.evaluate_population(var_archive, metric='test_acc', epoch=108)
        params, _ = self.evaluate_population(var_archive, metric='n_params')
        
        testerr = 1 - testacc
        params_norm = (params - self.max_min_metrics['min_params']) / (self.max_min_metrics['max_params'] - self.max_min_metrics['min_params'])

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
    
    
    
    