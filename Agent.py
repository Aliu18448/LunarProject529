#File to work on Agent
import tensorflow as tf
import pandas as pd
from keras import layers, Sequential

def neural_network(layers_sizes):
    model_layers = []
    
    for i, size in enumerate(layers_sizes):
        # Activation: ReLU for all but the last layer
        act = 'relu' if i < len(layers_sizes) - 1 else None
        
        model_layers.append(
            layers.Dense(
                units=size, 
                activation=act,
                kernel_initializer='glorot_uniform' # This is Xavier
            )
        )
        
    return Sequential(model_layers)
