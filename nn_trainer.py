import tensorflow as tf

import keras
from keras.src.models.sequential import Sequential
from keras import layers as klayers
from keras import optimizers as koptimizers
import numpy as np
from scipy.optimize import minimize

from typing import List, Callable

class trainer_model(Sequential):
    def __init__(self,
                 layers:klayers.Layer=None,
                 trainable:bool=True,
                 name:str=None,
                 ):
        super().__init__(layers=layers, trainable=trainable, name=name)

    # TODO: gradient based methods
    def back_minimize(self,
                 method = 'powell',):
        """
        After the model is trained, minimize the output by training the input.
        """

        # # @tf.function
        def to_minimize(x):
            pad_x = np.array([x])
            return self.predict(pad_x)

        x = np.random.rand(self.inputs[0].shape[1])
        
        result = minimize(to_minimize, x, method=method, tol=1e-6)

        return result


        # Optimize with gradients
        x = tf.Variable([list(x)])

        step = 200
        for _ in range(step):
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = self.predict([x])[0]
            grad = tape.gradient(y, x)
            optimizer.apply_gradients(zip(grad, x))
        
        return x

    @staticmethod
    def default_model(input_shape:tuple):
        return trainer_model(
            layers=[
                klayers.Input(input_shape),
                klayers.Dense(64, activation='relu', input_shape=(2,),
                                   kernel_initializer='random_normal',
                                   kernel_regularizer=keras.regularizers.l2(0.01)),
                klayers.Dense(32, activation='relu', input_shape=(2,)),
                klayers.Dense(12, activation='relu', input_shape=(2,)),
                klayers.Dense(1, activation='sigmoid', input_shape=(2,)),
                ],
                name='default_model'
            )
    