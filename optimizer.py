from typing import Any
from scipy.optimize import minimize
from nn_model import NNModel
import numpy as np

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
        
        if method == 'Neural Network':
            return self.NN_opt(min_func, x0, callback=callback)
        else:
            return minimize(min_func, x0, method=method,callback=callback)

    def NN_opt(self,func, para_size, callback=None):
        # optimize using neural network
        sample_x = np.zeros([0,para_size])
        sample_y = np.array([])

        optimal = [None,1]

        for _ in range(20) : # blackhole register
            para = np.random.uniform(-np.pi/2, np.pi/2, para_size)
            y = func(para)

            if y < optimal[1]:
                optimal = [para, y]
            sample_x = np.append(sample_x, [para], axis=0)
            sample_y = np.append(sample_y, y)

        nn_model = NNModel(para_size,layer_sizes=[32,1],activation_functions=['relu','sigmoid'])
        for i in range(20):
            nn_model.fit(sample_x, sample_y,epochs=100, verbose=0)
            prediction = nn_model.predict_optimum()
            y = func(prediction)

            if y < optimal[1]:
                optimal = [prediction, y]
            sample_x = np.append(sample_x, [prediction], axis=0)
            sample_y = np.append(sample_y, y)

        return optimal


