from typing import Any
from scipy.optimize import minimize

class Optimizer:

    _available_methods = ['Neural Network', 'Nelder-Mead', 'Powell', 'CG', 'BFGS']

    @staticmethod
    def list_methods():
        return Optimizer._available_methods

    def __init__(self, method:str='Neural Network') -> None:
        self.method = method
        
        self.saved_path = None
    
    @property
    def get_path_x(self):
        if self.saved_path is None:
            return None
        else:
            return self.path_x
    @property
    def get_path_y(self):
        if self.saved_path is None:
            return None
        else:
            return self.path_y

    def optimize(self, func, x0, callback=None, record_path=True, method:str=None) -> Any:

        if record_path:
            self.path_x = []
            self.path_y = []
            def min_func(x):
                self.path_x.append(x)
                y = func(x)
                self.path_y.append(y)
                return y
        else:
            min_func = func

        if method is None:
            method = self.method

        if method not in Optimizer._available_methods:
            raise ValueError(f'Optimizer method {method} not available. Available methods are {self.list_methods()}')
        
        # TODO: Implement NN method
        if method == 'Neural Network':
            pass
            raise NotImplementedError('Neural Network method not implemented yet')
            return None
        else:
            return minimize(min_func, x0, method=method,callback=callback)
