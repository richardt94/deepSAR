import numpy as np

#rescale and demean the input images according to their mean/std
#deviation
def standardise(x):

    smean = np.mean(x)
    sstd = np.std(x)

    x = (x - smean)/sstd

    return x
