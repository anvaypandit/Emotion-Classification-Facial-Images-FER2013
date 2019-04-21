import numpy as np
import pandas as pd
from tensorflow.python.lib.io import file_io
from skimage.transform import resize
from keras import backend as K
from keras.utils import to_categorical
from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import Flatten, Dense 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, Callback,ModelCheckpoint
import h5py # For saving the model

# PARAMETERS ##########################################################################################################################################
# Size of the images
img_height, img_width = 197, 197

# Parameters
num_classes         = 2     # ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# MODEL ###############################################################################################################################################

# Create the based on ResNet-50 architecture pre-trained model
    # model:        Selects one of the available architectures vgg16, resnet50 or senet50
    # include_top:  Whether to include the fully-connected layer at the top of the network
    # weights:      Pre-training on VGGFace
    # input_shape:  Optional shape tuple, only to be specified if include_top is False (otherwise the input
    #               shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with
    #               'channels_first' data format). It should have exactly 3 inputs channels, and width and
    #               height should be no smaller than 197. E.g. (200, 200, 3) would be one valid value.
# Returns a keras Model instance
base_model = VGGFace(
    model       = 'vgg16',
    include_top = False,
    weights     = 'vggface',
    input_shape = (img_height, img_width, 3))

# Places x as the output of the pre-trained model
x = base_model.output

# Flattens the input. Does not affect the batch size
x = Flatten()(x)

# Add a fully-connected layer and a logistic layer
# Dense implements the operation: output = activation(dot(input, kernel) + bias(only applicable if use_bias is True))
    # units:        Positive integer, dimensionality of the output space
    # activation:   Activation function to use
    # input shape:  nD tensor with shape: (batch_size, ..., input_dim)
    # output shape: nD tensor with shape: (batch_size, ..., units)

#x = Dense(1024, activation = 'relu')(x)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

test_datagen = ImageDataGenerator(
    samplewise_center=True,  # set each sample mean to 0
    rescale=(1/255.),
    data_format='channels_last')

test_generator = test_datagen.flow_from_directory(
        'data/Test',
        target_size=(197,197),
        class_mode='categorical',
        batch_size=1,
        shuffle=False)

pred = []
# Load all models and set the trained weights
model = Model(inputs = base_model.input, outputs = predictions)
model.load_weights('./emotion0/Model.06-0.9050.hdf5')

# Save predictions in pred array
test_generator.reset()
pred.append(model.predict_generator(test_generator, steps=3589, verbose=1))

model = Model(inputs = base_model.input, outputs = predictions)
model.load_weights('./emotion1/Model.05-0.9929.hdf5')

test_generator.reset()
pred.append(model.predict_generator(test_generator, steps=3589, verbose=1))

model = Model(inputs = base_model.input, outputs = predictions)
model.load_weights('./emotion2/Model.12-0.8837.hdf5')

test_generator.reset()
pred.append(model.predict_generator(test_generator, steps=3589, verbose=1))

model = Model(inputs = base_model.input, outputs = predictions)
model.load_weights('./emotion3/Model.03-0.9308.hdf5')

test_generator.reset()
pred.append(model.predict_generator(test_generator, steps=3589, verbose=1))

model = Model(inputs = base_model.input, outputs = predictions)
model.load_weights('./emotion4/Model.07-0.8562.hdf5')

test_generator.reset()
pred.append(model.predict_generator(test_generator, steps=3589, verbose=1))

model = Model(inputs = base_model.input, outputs = predictions)
model.load_weights('./emotion5/Model.13-0.9589.hdf5')

test_generator.reset()
pred.append(model.predict_generator(test_generator, steps=3589, verbose=1))

model = Model(inputs = base_model.input, outputs = predictions)
model.load_weights('./emotion6/Model.03-0.8811.hdf5')

test_generator.reset()
pred.append(model.predict_generator(test_generator, steps=3589, verbose=1))


# Calculate accuracy
acc = 0
for i in range(len(test_generator.classes)):
  
  print("Target:", test_generator.classes[i])
  max_index = -1
  max = -1
  
  for emotion in range(7):
    if pred[emotion][i][0] > max:
      max_index = emotion
      max = pred[emotion][i][0]

  print("Prediction:", max_index)

  if max_index == test_generator.classes[i]:
    acc += 1

print("Accuracy: ", acc/len(test_generator.classes))