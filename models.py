import numpy as np
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.activations import softmax

from noise_layer import speckle_noise

#a simple denoising autoencoder as a Keras Model object. The architecture is influenced by
#the denoising autoencoder example at https://keras.io/examples/mnist_denoising_autoencoder/

def denoise_autoenc(pretrained_weights = None, input_size = (256,256,1), batch_size = 128,
    kernel_size = 3, latent_dim = 16, layer_filters = [32,64]):
    #input layer
    inputs = Input(input_size)
    #encoder layers (a CNN)
    x = inputs
    for filters in layer_filters:
        #padding must be 'same' for the decoder stage to work
        x = Conv2D(filters = filters,
           kernel_size = kernel_size,
           strides = 2,
           activation = 'relu',
           padding = 'same')(x)
    
    #shape of the bottom encoder layer
    shape = K.int_shape(x)

    #Dense layer provides the final encoded representation
    x = Flatten()(x)
    latent = Dense(latent_dim)(x)

    encoder = Model(inputs,latent)

    #the decoder model reverses the encoder layers
    latent_inputs = Input(shape = (latent_dim,))
    x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters = filters,
                            kernel_size = kernel_size,
                            strides = 2,
                            activation = 'relu',
                            padding = 'same')(x)

    
    #final deconv layer restores the size of original input
    x = Conv2DTranspose(filters = 1,
                        kernel_size = kernel_size,
                        padding = 'same')(x)
    
    outputs = Activation('sigmoid')(x)

    decoder = Model(latent_inputs,outputs)

    autoenc = Model(inputs, decoder(encoder(inputs)))

    #get it ready for training
    autoenc.compile(loss='mse', optimizer='adam')

    return autoenc
    

    
