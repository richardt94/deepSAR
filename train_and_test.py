from denoising_models import denoise_autoenc
from noise_layer import speckle_noise

from keras.preprocessing.image import ImageDataGenerator

#generator will apply speckle noise to each image in real time. When we test on actual SAR images
#this will not happen obvs
datagen = ImageDataGenerator(preprocessing_function = speckle_noise)

train_iter = datagen.flow_from_directory("/g/data/r78/rlt118/UCMerced_LandUse/Images/train",
                                            color_mode = 'grayscale', batch_size = 128,
                                            class_mode = 'input',
                                            save_to_dir="/g/data/r78/rlt118/UCMerced_LandUse/Images/train/noisy")


val_iter = datagen.flow_from_directory("/g/data/r78/rlt118/UCMerced_LandUse/Images/test",
                                            color_mode = 'grayscale', batch_size = 128,
                                            class_mode = 'input',
                                            save_to_dir="/g/data/r78/rlt118/UCMerced_LandUse/Images/test/noisy")


                                            
#initialise and fit the model
model = denoise_autoenc()

model.fit_generator(train_iter, steps_per_epoch = len(train_iter), epochs = 20,
                    validation_data = val_iter, validation_steps = len(val_iter))


