import argparse
from tqdm import tqdm

from algorithms.nsga2 import NSGAII
from algorithms.novelty_search.ns_nas_bench_201 import NoveltySearch201
from algorithms.novelty_search.ns_nas_bench_101 import NoveltySearch101
from algorithms.novelty_search.ns_nas_bench_1shot1 import NoveltySearch1Shot1
from algorithms.novelty_search.ns_nas_bench_asr import NoveltySearchASR
from algorithms.novelty_search.ns_trans_nas_bench_101 import NoveltySearchTrans101



def parse_arguments():
    parser = argparse.ArgumentParser("Novelty Search Neural Architecture Search")
    parser.add_argument('--performance_criteria', type=str, default='synflow', help='performance objective [synflow, jacov, snip, val_acc, synflow-jacov-snip,...]')
    parser.add_argument('--n_runs', type=int, default=30, help='number of runs')
    parser.add_argument('--pop_size', type=int, default=20, help='population size')
    parser.add_argument('--n_gen', type=int, default=50, help='number of generations')
    parser.add_argument('--continue_run', type=int, default=0, help='continue run')
    parser.add_argument('--problem_name', type=str, default='NASBench201', help='Name of problems [NASBench101, NASBench201, NASBench1Shot1]')
    parser.add_argument('--method', type=str, default='TF-PDNS', help='name of method [MOENAS, TF-MOENAS, PDNS, TF-PDNS]')
    parser.add_argument('--dataset', type=str, default=None, help='name of dataset (only available of NAS-Bench-201) [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--save_result', type=bool, default=True, help='save result or not')
    parser.add_argument('--search_space', type=int, default=None, help='[1, 2, 3] for NAS-Bench-1Shot1')
    

    args = parser.parse_args()
    if args.problem_name=='NASBench201':
        args.n_var=6
        if args.performance_criteria=='synflow-jacov':
            args.objective_names = ['synflow', 'jacov', 'flops']
            args.n_objs = 3
        elif args.performance_criteria=='synflow-jacov-snip':
            args.objective_names = ['synflow', 'jacov', 'snip', 'flops']
            args.n_objs = 4
        else:  
            args.objective_names = [args.performance_criteria, 'flops']
            args.n_objs = 2
        
    elif args.problem_name=='NASBench101':
        args.n_var=26
        if args.performance_criteria=='synflow-jacov':
            args.objective_names = ['synflow', 'jacob_cov', 'n_params']
            args.n_objs = 3
        elif args.performance_criteria=='synflow-jacov-snip':
            args.objective_names = ['synflow', 'jacob_cov', 'snip', 'n_params']
            args.n_objs = 4
        else:
            args.objective_names = [args.performance_criteria, 'n_params']
            args.n_objs = 2
            
    elif args.problem_name=='NASBench1Shot1':
        if args.search_space==1:
            args.n_var=8
        elif args.search_space==2:
            args.n_var=9
        elif args.search_space==3:
            args.n_var=10
            
        if args.performance_criteria=='synflow-jacov':
            args.objective_names = ['synflow', 'jacob_cov', 'n_params']
            args.n_objs = 3
        elif args.performance_criteria=='synflow-jacov-snip':
            args.objective_names = ['synflow', 'jacob_cov', 'snip', 'n_params']
            args.n_objs = 4
        else:
            args.objective_names = [args.performance_criteria, 'n_params']
            args.n_objs = 2
    return args

if __name__ == '__main__':
    args = parse_arguments()
    print(args.method)
    for seed in range(args.continue_run, args.n_runs):
        print('seed:', seed)
        if args.method in ['MOENAS', 'TF-MOENAS']:
            algo = NSGAII(seed=seed, **vars(args)) 
            algo.run()
        elif args.method in ['PDNS', 'TF-PDNS']:
            if args.problem_name=='NASBench201':
                algo = NoveltySearch201(seed=seed, **vars(args))
            elif args.problem_name=='NASBench101':
                algo = NoveltySearch101(seed=seed, **vars(args))
            elif args.problem_name=='NASBench1Shot1':
                algo = NoveltySearch1Shot1(seed=seed, **vars(args))
            elif args.problem_name=='NASBenchASR':
                algo = NoveltySearchASR(seed=seed, **vars(args))
            elif args.problem_name=='TransNASBench101':
                algo = NoveltySearchTrans101(seed=seed, **vars(args))
            algo.run()
