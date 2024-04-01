import torch
import os
import pickle
import time

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.core.callback import Callback

from algorithms.base_algo import BaseAlgo
from problems.nas_bench_201 import NASBench201
from problems.nas_bench_101 import NASBench101
from problems.nas_bench_1shot1 import NASBench1Shot1
from problems.nas_bench_asr import NASBenchASR
from problems.trans_nas_bench_101 import TransNASBench101



from utils.utils_nas_bench_201 import *
from utils.utils_nas_bench_101 import *
from utils.utils_nas_bench_1shot1 import *
from utils.utils import *

class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["var_pop"] = []
        self.data["obj_pop"] = []
        self.data["var_archive"] = []
        self.data["obj_archive"] = []
        self.data["test_var_archive"] = []
        self.data["test_obj_archive"] = []
        self.data["indicators"] = []
        self.data["log_quality"] = []
        self.data["time"] = None


    def notify(self, algorithm):
        var_pop = algorithm.pop.get("X")
        obj_pop = algorithm.pop.get("F")


        var_archive = self.data["var_archive"][-1].copy() if self.data["var_archive"] else []
        obj_archive = self.data["obj_archive"][-1].copy() if self.data["obj_archive"] else []

        if algorithm.problem.problem_name=='NASBench201':
            for var_ind, obj_ind in zip(var_pop, obj_pop):
                var_archive, obj_archive = update_elitist_archive(var_archive, obj_archive, var_ind, obj_ind)

                test_var_archive_dict, test_obj_archive_dict, val_var_archive_dict, val_obj_archive_dict, indicators = {}, {}, {}, {}, {}
                for dataset in algorithm.problem.datasets:
                    test_var_archive, test_obj_archive = algorithm.problem._evaluate_and_filter_true_archive(var_archive, dataset)
                    test_var_archive_dict[dataset] = test_var_archive
                    test_obj_archive_dict[dataset] = test_obj_archive

                    pareto_front_testacc = np.array(algorithm.problem.pareto_fronts_testacc[dataset])
                                        
                    indicators[dataset] = algorithm.problem._calc_performance_indicators(test_obj_archive, pareto_front_testacc)

                self.data["indicators"].append(indicators)
                self.data["var_archive"].append(var_archive)
                self.data["obj_archive"].append(obj_archive)
                self.data["test_var_archive"].append(test_var_archive_dict)
                self.data["test_obj_archive"].append(test_obj_archive_dict)
        else:
            for var_ind, obj_ind in zip(var_pop, obj_pop):
                var_archive, obj_archive = update_elitist_archive(var_archive, obj_archive, var_ind, obj_ind)
                if algorithm.problem.problem_name=='TransNASBench101':
                    test_var_archive, test_obj_archive = algorithm.problem._evaluate_and_filter_true_archive(var_archive, algorithm.problem.dataset)
                else:
                    test_var_archive, test_obj_archive = algorithm.problem._evaluate_and_filter_true_archive(var_archive)
                if algorithm.problem.problem_name=='TransNASBench101':
                    pareto_fronts = algorithm.problem.pareto_fronts[algorithm.problem.dataset]
                else:
                    pareto_fronts = algorithm.problem.pareto_fronts
                indicators = algorithm.problem._calc_performance_indicators(test_obj_archive, pareto_fronts)

                self.data["indicators"].append(indicators)
                self.data["var_archive"].append(var_archive)
                self.data["obj_archive"].append(obj_archive)
                self.data["test_var_archive"].append(test_var_archive)
                self.data["test_obj_archive"].append(test_obj_archive)

        self.data["var_pop"].append(var_pop)
        self.data["obj_pop"].append(obj_pop)
        self.data["time"] = algorithm.problem.time
        self.data["log_archs"] = algorithm.problem.log_archs


class NSGAII(BaseAlgo):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)        
        
    def run(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        stop_criteria = ('n_gen', self.n_gen)
        
        if self.problem_name=='NASBench101':
            initial_pop = initialize_population(self.pop_size)
            algorithm = NSGA2(pop_size=self.pop_size,
                            sampling=initial_pop,
                            crossover=CustomTwoPointCrossover(prob=0.9),
                            mutation=CustomPolynomialMutation(prob=1.0/self.n_var, eta=1.0, repair=RoundingRepair()),
                            eliminate_duplicates=True)
        elif self.problem_name in ['NASBench201', 'TransNASBench101', 'NASBench1Shot1', 'NASBenchASR']:
            algorithm = NSGA2(pop_size=self.pop_size,
                            sampling=IntegerRandomSampling(),
                            crossover=TwoPointCrossover(prob=1.0),
                            mutation=PolynomialMutation(prob=1.0/self.n_var, eta=1.0, repair=RoundingRepair()),
                            eliminate_duplicates=True)       
        
        if self.problem_name=='NASBench201':
            problem = NASBench201(**self.kwargs)
        elif self.problem_name=='NASBench101':
            problem = NASBench101(**self.kwargs)
        elif self.problem_name=='NASBench1Shot1':
            problem = NASBench1Shot1(**self.kwargs)
        elif self.problem_name=='NASBenchASR':
            problem = NASBenchASR(**self.kwargs)
        elif self.problem_name=='TransNASBench101':
            problem = TransNASBench101(**self.kwargs)
            
        results = minimize(
            problem = problem,
            algorithm = algorithm,
            seed = self.seed,
            callback=MyCallback(),
            termination = stop_criteria
        )
        
        data = results.algorithm.callback.data
        
        algo_time = time.time()
        data['time'].append(algo_time + results.algorithm.problem.cum_time)
        if self.save_result:
            if self.problem_name in ['NASBench201', 'TransNASBench101']:
                save_dir = f'results/{self.problem_name}/{self.method}/{self.performance_criteria}/{self.dataset}/'
            elif  self.problem_name in ['NASBench1Shot1']:
                save_dir = f'results/{self.problem_name}/{self.method}/{self.performance_criteria}/{self.search_space}/'
            else:
                save_dir = f'results/{self.problem_name}/{self.method}/{self.performance_criteria}'
            
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f'seed_{self.seed}.pkl'), 'wb') as f:
                pickle.dump(data, f)

