import numpy as np

from keras.layers import Layer

from keras import backend as K

#mess up the input with multiplicative 'speckle' noise
def speckle_noise(x, train=False):
    if not train:
        return x

    noise = K.random_uniform(K.shape(x),minval=0.0001)
    
    #speckle noise is exponentially distributed for a single-look intensity image
    speck = -K.log(noise)

    return x*speck
