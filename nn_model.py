import tensorflow as tf
import keras
from keras.src.models.sequential import Sequential
import numpy as np
from scipy.optimize import minimize

from typing import List, Callable

class trainer_model(Sequential):
    def __init__(self,
                 layers:keras.layers.Layer=None,
                 trainable:bool=True,
                 name:str=None,
                 ):
        super().__init__(layers=layers, trainable=trainable, name=name)
    
    @staticmethod
    def default_model(input_shape:tuple):
        return trainer_model(
            layers=[
                keras.layers.Input(input_shape),
                keras.layers.Dense(64, activation='relu', input_shape=(2,),
                                   kernel_initializer='random_normal',
                                   kernel_regularizer=keras.regularizers.l2(0.01)),
                keras.layers.Dense(32, activation='relu', input_shape=(2,)),
                keras.layers.Dense(12, activation='relu', input_shape=(2,)),
                keras.layers.Dense(1, activation='sigmoid', input_shape=(2,)),
                ],
                name='default_model'
            )
    
    # TODO: add optimizer
    def optimize():
        """
        After the model is trained, minimize the output by training the input.
        """
        pass

class NNModel:

    def __init__(self,
                 input_shape:tuple,
                 layers:List[keras.layers.Layer] | List[tuple],
                 initializer:keras.initializers.Initializer=None,
                 regulizer:keras.regularizers.Regularizer=None,):
        """
        construct a Keras Sequential model with the given parameters.
        
        parameters:
        input_shape - shape of the input layer. exp: `(5,)`
        """

        self.input_shape = input_shape

        self.initializer = initializer
        self.regulizer = regulizer

        if isinstance(layers[0], keras.layers.Layer):
            self.layers = layers
        elif isinstance(layers[0], tuple):
            self.layers = [keras.layers.Dense(
                size,func,
                kernel_initializer=self.initializer,
                kernel_regularizer=self.regulizer)
                for size, func in layers]
        

        self.create_model()
    
    def create_model(self):
        # Initiate model with input layer
        self.model = Sequential(
            [
                keras.layers.Input(shape=self.input_shape, dtype=tf.float32)
            ] + self.layers
        )

    def add_layer(self,
                  size:int,
                  act_func:Callable,
                  renew = False
                ) -> None:
        """
        Add a layer to the model.
        If renew is True, the model will be re-created.
        Otherwise, add the layer to the model without touch the previous layers.
        """

        self.layers.append(keras.layers.Dense(size, act_func))
        if renew:
            self.create_model()
        else:
            self.model.add(self.layers[-1])

    # TODO: fitting
    def fit(self, inputs, outputs,
            optimizer='Adam',
            loss:str='mean_squared_error',
            epochs:int=1,
            batch_size:int=None,
            validation_data=None,
            verbose=0):
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
