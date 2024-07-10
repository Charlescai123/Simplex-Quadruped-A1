import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Layer


class MLPModel(Model):
    def __init__(self, shape_input, shape_output, name='', output_activation=None):
        super(MLPModel, self).__init__()

        self.model = self.build_mlp_model(shape_input, shape_output, name, output_activation)

    # def __new__(cls, shape_input, shape_output, name='', output_activation=None):
    #     return cls.build_mlp_model(shape_input, shape_output, name, output_activation)

    @staticmethod
    def build_mlp_model(shape_input, shape_output, name='', output_activation=None):
        input = Input(shape=(shape_input,), name=name + 'input', dtype=tf.float16)
        dense1 = Dense(128, activation='relu', name=name + 'dense1')(input)
        dense2 = Dense(128, activation='relu', name=name + 'dense2')(dense1)
        dense3 = Dense(128, activation='relu', name=name + 'dense3')(dense2)
        output = Dense(shape_output, activation=output_activation, name=name + 'output')(dense3)
        model = Model(inputs=input, outputs=output, name=name)
        return model

    def call(self, inputs):
        return self.model(inputs)
