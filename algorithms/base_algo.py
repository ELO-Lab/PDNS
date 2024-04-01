import torch



class BaseAlgo():
    def __init__(self, seed, **kwargs):
        self.seed=seed 
        self.kwargs = kwargs
        
        for key in kwargs:
            setattr(self, key, kwargs[key])
    
    
    def _run(self):
        raise NotImplementedError


