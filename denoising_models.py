import numpy as np
from keras.models import *
from keras.layers import *

from keras import backend as K

from keras.optimizers import *

from noise_layer import speckle_noise

#a simple denoising autoencoder as a Keras Model object.

def denoise_autoenc(pretrained_weights = None, input_size = (64,64,1),
                     latent_dim = 8192):
    #input layer
    inputs = Input(input_size)

    shape = K.int_shape(inputs)

    x = Flatten()(inputs)
    latent = Dense(latent_dim, activation = None)(x)

    #the decoder model reverses the encoder layers
    x = Dense(shape[1]*shape[2]*shape[3],activation = None)(latent)
    outputs = Reshape((shape[1], shape[2], shape[3]))(x)


    autoenc = Model(inputs, outputs)

    #get it ready for training

    opt = Adam(lr=1e-2)

    autoenc.compile(loss='mean_squared_error', optimizer=opt)

    if(pretrained_weights):
    	autoenc.load_weights(pretrained_weights)

    return autoenc
    
#train flag tells the speckle_noise layer to actually add the synthetic noise. to be used for training
#or evaluation on simulated 'SAR images' from optical RGB.
#output_noisy tells the model to return the input with the speckle noise added (if applicable), in addition
#to the denoised output. This is useful for evaluation of the model both with the synthetic SAR speckle
#(if train = True) or real SAR images (if train=False) as it allows easier matching of an 

def conv_denoise_autoenc(pretrained_weights = None, input_size = (64,64,1), train = False, output_noisy=False):
    input_img = Input(input_size)

    noisy = Lambda(speckle_noise, arguments = {'train': train})(input_img)
    

    x = Conv2D(16,(3,3), activation='relu', padding='same')(noisy)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(16,(3,3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2,2), name='encoder')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='softsign', padding='same')(x)

    if output_noisy:
        autoencoder = Model(inputs=input_img, outputs=[decoded,noisy])
    else:
        autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    if(pretrained_weights):
    	autoencoder.load_weights(pretrained_weights)

    return autoencoder

    
