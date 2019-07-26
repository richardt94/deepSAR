from denoising_models import *
from standardisation import standardise

from keras.preprocessing.image import save_img

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import plot_model

#generator will apply speckle noise to each image in real time. When we test on actual SAR images
#this will not happen obvs
datagen = ImageDataGenerator(preprocessing_function = standardise)

val_iter = datagen.flow_from_directory("UCMerced_LandUse/Images/test",
                                            color_mode = 'grayscale', batch_size = 4,
                                            target_size = (256,256),
                                            class_mode = 'input')



#initialise and test the model
#we're still using the synthetic speckle so we need to set train=True
model = conv_denoise_autoenc(pretrained_weights='denoise_autoenc-60-0.21.hdf5', input_size=(256,256,1), train = True, output_noisy=True)

results = model.predict_generator(val_iter, steps = len(val_iter))

rfolder = 'test_results/'
ifolder = 'test_inputs/'

for indx,(result,noisy) in enumerate(zip(results[0],results[1])):

    fname = str(indx)+'.png'
    save_img(rfolder + fname, result)
    save_img(ifolder + fname, noisy)
