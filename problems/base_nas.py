from pymoo.core.problem import Problem
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
import time

class BaseNAS(Problem):
    def __init__(self, **kwargs):      
        for key in kwargs:
            setattr(self, key, kwargs[key])
        super().__init__(n_var=self.n_var, n_obj=self.n_objs)
        self.kwargs = kwargs
        self._n_gen = 0
        self._n_eval = 0
        self.ref_point = [1.05, 1.05]
        start_time = time.time()
        self.cum_time = 0
        self.time = [start_time]

    
    def _evaluate(self, var_pop, out, *args, **kwargs):
        raise NotImplementedError
        