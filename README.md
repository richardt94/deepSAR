# deepSAR
Deep learning for removing speckle noise from synthetic aperture radar (SAR) images

## Dependencies:
Python 3.7
Tensorflow r1.14

## Scripts:
train_and_test.py:
Trains the model on the modified UC Merced Land Use dataset, adding synthetic speckle to the input images
test_and_save.py:
Runs the model on the validation portion of the UC Merced dataset. Saves the synthetic noisy input and the results as png images.

## Yet to be implemented:
Workflow to test the model on real noisy SAR images obtained from a datacube (https://www.opendatacube.org/), specifically Digital Earth Australia.
