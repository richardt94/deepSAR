from denoising_models import *
from standardisation import standardise

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

#generator will apply speckle noise to each image in real time. When we test on actual SAR images
#this will not happen obvs
datagen = ImageDataGenerator(preprocessing_function = standardise)

train_iter = datagen.flow_from_directory("UCMerced_LandUse/Images/train",
                                            color_mode = 'grayscale', batch_size = 4,
                                            target_size = (256,256),
                                            class_mode = 'input')


val_iter = datagen.flow_from_directory("UCMerced_LandUse/Images/test",
                                            color_mode = 'grayscale', batch_size = 4,
                                            target_size = (256,256),
                                            class_mode = 'input')


                                            
#initialise and fit the model
model = conv_denoise_autoenc(input_size=(256,256,1), train = True)

filepath="denoise_autoenc-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callback_list = [checkpoint]

model.fit_generator(train_iter, steps_per_epoch = len(train_iter), epochs = 60,
                    validation_data = val_iter, validation_steps = len(val_iter),
                    callbacks = callback_list)


