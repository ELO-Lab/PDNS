import pickle
import json
import time
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from problems.base_nas import BaseNAS
from utils.utils_nas_bench_201 import *
from utils.utils import *


class NASBench201(BaseNAS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.datasets = ['cifar10', 'cifar100', 'ImageNet16-120'] if self.dataset=='cifar10' else [self.dataset]
        self.database, self.zc_database, self.pareto_fronts_testacc, self.pareto_fronts_valacc, self.max_min_metrics = self._load_database()
        self.xl = np.ones(self.n_var) * 0
        self.xu = np.ones(self.n_var) * 4
        self.log_archs = []
        
            
                
    def _load_database(self):
        database, pareto_fronts_valacc = {}, {}      
        max_min_metrics = pickle.load(open('data/NAS_Bench_201/max_min_measures.pickle', 'rb'))
        for dataset in self.datasets:
            database[dataset] = pickle.load(open(f'data/NAS_Bench_201/[{dataset}]_database.p', 'rb'))
            pareto_fronts_valacc[dataset] = pickle.load(open(f'data/NAS_Bench_201/[POF_ValAcc_FLOPs]_[NAS201_{dataset}].p', 'rb'))
            if dataset != 'ImageNet16-120':
                pareto_fronts_valacc[dataset][:, 0] /= 100
            pareto_fronts_valacc[dataset][:, 1] = (pareto_fronts_valacc[dataset][:, 1] - max_min_metrics[dataset]['flops_min']) / (max_min_metrics[dataset]['flops_max'] - max_min_metrics[dataset]['flops_min'])
            
        zc_database = json.load(open('data/NAS_Bench_201/zc_nasbench201.json', 'r'))
        
        pareto_fronts_testacc = json.load(open('data/NAS_Bench_201/nb201_pf_norm.json', 'r'))
        
        return database, zc_database, pareto_fronts_testacc, pareto_fronts_valacc, max_min_metrics
    
    def _calc_performance_indicators(self, elitist_archive, pareto_fronts_testacc, elitist_archive_valacc=None, pareto_fronts_valacc=None):
        get_igd = IGD(pareto_fronts_testacc)
        get_igd_plus = IGDPlus(pareto_fronts_testacc)
        get_hv = HV(ref_point=self.ref_point)        

        indicators = {}
        indicators['igd'] = get_igd(elitist_archive)
        indicators['igd_plus'] = get_igd_plus(elitist_archive)
        indicators['hv'] = get_hv(elitist_archive)
        

        return indicators
    
    def _evaluate_and_filter_true_archive(self, var_archive, dataset):
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

    def _evaluate(self, var_pop, out, *args, **kwargs):

        all_scores = np.zeros((len(var_pop), self.n_objs))
        for i, var_ind in enumerate(var_pop):
            for j, metric in enumerate(self.objective_names):
                if metric in ['synflow', 'jacov', 'snip', 'fisher', 'val_acc']:
                    score, performance_time = self.evaluate_arch(var_ind, metric=metric, dataset=self.dataset, epoch=12)
                    score *= -1
                    all_scores[i][j] = score
                elif metric in ['flops']:
                    score, complexity_time = self.evaluate_arch(var_ind, metric=metric, dataset=self.dataset)
                    all_scores[i][j] = score
            algo_time = time.time()
            
            if len(self.log_archs)==0  or var_ind.tolist() not in self.log_archs:
                    self.cum_time += performance_time + complexity_time
                    self.log_archs.append(var_ind.tolist())      
            self.time.append(algo_time + self.cum_time)
                                
        self._n_gen += 1
        out["F"] = all_scores
