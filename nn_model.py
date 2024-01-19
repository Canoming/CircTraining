'''
Author: Guowenyu
Date: 2024-01-16 15:42:44
LastEditTime: 2024-01-19 13:45:45
LastEditors: Sleepy
Description: In User Settings Edit
FilePath: /lq/NN_compilation/neural_netwoek_learning/nn_model.py
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy.optimize import minimize


class NNModel:
    """
    Neural Network model class.

    Given a list of date: `[[para, fidelity] ...]`, train a neural network model to predict the optimum of the given function.

    """
    
    def __init__(self, input_shape, layer_sizes, activation_functions, seed=None):
        """
        construct a Keras Sequential model with the given parameters.
        
        parameters:
        input_shape - shape of the input data.
        layer_sizes - list of each layer's size.
        activation_functions - list of each layer's activation function.
        seed - random seed for the model.
        """
        
        self.input_shape = input_shape
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.seed = seed
        self.model = self.create_nn_model(input_shape, layer_sizes, activation_functions, seed)


    def create_nn_model(self,input_shape, layer_sizes, activation_functions, seed=None):
        """
        construct a Keras Sequential model with the given parameters.
        
        parameters:
        input_shape - shape of the input data.
        layer_sizes - list of each layer's size.
        activation_functions - list of each layer's activation function.
        seed - random seed for the model.
        
        returns:
        model of  Keras Sequential
        """
        # make sure the number of layers and activation functions match
        if len(layer_sizes) != len(activation_functions):
            raise ValueError("number of layers and activation functions must match")

        # assert len(layer_sizes) == len(activation_functions), "number of layers and activation functions must match"

        # initialize random number generator
        if seed is not None:
            tf.random.set_seed(seed)
        initializer = tf.keras.initializers.he_normal(seed=seed)

        # construct the model
        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=input_shape, dtype=tf.float32))

        # add hidden layers
        for size, activation in zip(layer_sizes, activation_functions):
            model.add(keras.layers.Dense(size,
                                        kernel_initializer=initializer,
                                        kernel_regularizer=keras.regularizers.l2(1e-8),
                                        activation=activation))

        return model


    # TODO: fitting
    def fit(self, inputs, outputs, optimizer='Adam', loss:str='mean_squared_error', epochs:int=1, batch_size:int=None, validation_data=None, verbose=1):
        """
        fit the model with the given parameters.

        parameters:
        x - input data, parameters.
        y - target data, fidelity.
        optimizer - optimizer used for training.
        loss - loss function used for training.
            ???
    
        epochs - number of epochs.
        batch_size - batch size.
        validation_data - validation data.
        verbose - verbose mode.
        
        """
        self.inputs = inputs
        self.outputs = outputs
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['mean_squared_error'])
        self.model.fit(x=self.inputs, # parameters
                       y=self.outputs, # data
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=validation_data,
                       verbose=verbose)

        
    # TODO: backward search
    # def predict_optimum(self, use_nn, restarts=1, completely_random_start=False):
    #     """
    #     predict the optimum of the given function using the model.
        
    #     parameters:
    #     use_nn - whether to use the model for prediction.
    #     restarts - number of restarts.
    #     """

    #     self.model.predict()
    def predict(self, inputs):
        """
        Use the neural network model to predict the output for given parameters.
        
        Parameters:
        params - A list or array of parameters.
        
        Returns:
        Predicted output from the neural network.
        """
        
        # params = np.array(params).reshape((1, -1))
    
        predictions = self.model.predict(inputs)
        print(predictions)
        return predictions.flatten()  # 确保输出是一维数组

        # return self.model.predict(params)[0][0]

    def predict_optimum(self, restarts=1, completely_random_start=False):
        """
        Predict the optimum of the function using the model.
        
        Parameters:
        restarts - The number of times to restart the optimization process.
        completely_random_start - Whether to start optimization from random points.
        
        Returns:
        The parameters that yield the optimum prediction.
        """

        # 定义要优化的函数
        def objective(inputs):
            return self.predict(inputs)

        # 设置初始参数
        if completely_random_start:
            init_inputs = np.random.rand(self.input_shape[0])
        else:
            # 如果不是完全随机，可以从训练数据中选择一个点
            init_inputs = self.inputs[0]

        # 进行优化
        opt_inputs = None
        opt_outputs = np.inf
        for _ in range(restarts):
            res = minimize(objective, init_inputs, method='L-BFGS-B')


            if res.fun < opt_outputs:
                opt_outputs = res.fun
                opt_inputs = res.x

        return opt_inputs
        

        
    
    # def nn_optimizer_func(optimizer_name, lr, beta_1, beta_2, epsilon=None, decay=0.0, amsgrad=False):
    #     return tf.keras.optimizers.optimizer_name(
    #         lr=lr,
    #         beta_1=beta_1,
    #         beta_2=beta_2,
    #         epsilon= epsilon,
    #         decay=decay,
    #         amsgrad=amsgrad)
    
    def nn_optimizer_func(self, optimizer_name="Adam", lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False):
        optimizers = {
            # optimizer class here.
            'Adam': tf.keras.optimizers.Adam,
            'SGD': tf.keras.optimizers.SGD,
            # can add more optimizers here
        }

        optimizer_class = optimizers.get(optimizer_name)
        if not optimizer_class:
            raise ValueError(f"Optimizer '{optimizer_name}' not recognized")

        return optimizer_class(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)







# test the function
if __name__ == "__main__":

    example_model = NNModel(input_shape=(5,), 
                                        layer_sizes=[96, 64, 10, 1], 
                                        activation_functions=[tf.nn.elu, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid],
                                        seed=42)
    # example_model.summary()

    optimizer = example_model.nn_optimizer_func(optimizer_name="Adam", lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    x = [[1,2,3,4,5],[1,2,3,4,6]] # parameters
    y = [[1],[3]] #fidelities
    example_model.fit(x, y, optimizer, loss='mean_squared_error', epochs=1, batch_size=None, validation_data=None, verbose=1)
    example_model.predict_optimum(restarts=1, completely_random_start=False)
