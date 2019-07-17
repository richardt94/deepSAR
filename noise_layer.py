import numpy as np

#mess up the input with multiplicative 'speckle' noise
def speckle_noise(x):
    noise = np.random.uniform(size=x.shape)

    #speckle noise is exponentially distributed for a single-look intensity image
    speck = -np.log(noise)

    return np.multiply(x,speck)
