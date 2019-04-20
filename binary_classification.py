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
epochs_top_layers   = 5
epochs_all_layers   = 15
batch_size          = 64

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

train_datagen = ImageDataGenerator(
    samplewise_center=True,  # set each sample mean to 0
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    shear_range=10, # 10 degrees
    zoom_range=0.1,
    fill_mode='reflect',
    rescale=(1/255.),
    horizontal_flip=True,  # randomly flip images
    data_format='channels_last',
    vertical_flip=False)  # randomly flip images

test_datagen = ImageDataGenerator(
    samplewise_center=True,  # set each sample mean to 0
    rescale=(1/255.),
    data_format='channels_last')


def scheduler(epoch):
    updated_lr = K.get_value(model.optimizer.lr) * 0.5
    if (epoch % 19 == 0) and (epoch != 0):
        K.set_value(model.optimizer.lr, updated_lr)
        print(K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)

# Learning rate scheduler
    # schedule: a function that takes an epoch index as input (integer, indexed from 0) and current learning
    #           rate and returns a new learning rate as output (float)
reduce_lr = LearningRateScheduler(scheduler)

# Reduce learning rate when a metric has stopped improving
  # monitor: 	Quantity to be monitored
  # factor: 	Factor by which the learning rate will be reduced. new_lr = lr * factor
  # patience:	Number of epochs with no improvement after which learning rate will be reduced
  # mode: 	One of {auto, min, max}
  # min_lr:	Lower bound on the learning rate
reduce_lr_plateau = ReduceLROnPlateau(
  monitor 	= 'val_loss',
  factor		= 0.5,
  patience	= 3,
  mode 		= 'auto',
  min_lr		= 1e-8)

# Stop training when a monitored quantity has stopped improving
# monitor:		Quantity to be monitored
  # patience:		Number of epochs with no improvement after which training will be stopped
  # mode: 		One of {auto, min, max}
early_stop = EarlyStopping(
  monitor 	= 'val_acc',
  patience 	= 15,
  mode 		= 'auto')

for i in range(5, 7):
  print("Emotion: ", i)

  filepath='emotion' + str(i) + '/Model.{epoch:02d}-{val_acc:.4f}.hdf5'
  checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

  # The model we will train
  model = Model(inputs = base_model.input, outputs = predictions)
  # model.summary()
  
  train_generator = train_datagen.flow_from_directory(
          'data/Train' + str(i),
          target_size=(197,197),
          batch_size=batch_size,
          class_mode='categorical')

  validation_generator = test_datagen.flow_from_directory(
          'data/Valid' + str(i),
          target_size=(197,197),
          batch_size = batch_size,
          class_mode='categorical')

  # UPPER LAYERS TRAINING ###############################################################################################################################

  # First: train only the top layers (which were randomly initialized) freezing all convolutional ResNet-50 layers
  for layer in base_model.layers:
      layer.trainable = False

  # Compile (configures the model for training) the model (should be done *AFTER* setting layers to non-trainable)
      # optimizer:    String (name of optimizer) or optimizer object
          # lr:       Float >= 0. Learning rate
          # beta_1:   Float, 0 < beta < 1. Generally close to 1
          # beta_2:   Float, 0 < beta < 1. Generally close to 1
          # epsilon:  Float >= 0. Fuzz factor
          # decay:    Float >= 0. Learning rate decay over each update
      # loss:     String (name of objective function) or objective function
      # metrics:  List of metrics to be evaluated by the model during training and testing
  model.compile(
      optimizer   = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0), 
      loss        = 'binary_crossentropy', 
      metrics     = ['accuracy'])


  # Train the model on the new data for a few epochs (Fits the model on data yielded batch-by-batch by a Python generator)
      # generator:        A generator or an instance of Sequence (keras.utils.Sequence) object in order to avoid duplicate data when using multiprocessing
      #                   The output of the generator must be either {a tuple (inputs, targets)} {a tuple (inputs, targets, sample_weights)}
      # steps_per_epoch:  Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch
      #                   It should typically be equal to the number of unique samples of your dataset divided by the batch size 
      # epochs:           Integer, total number of iterations on the data
      # validation_data:  This can be either {a generator for the validation data } {a tuple (inputs, targets)} {a tuple (inputs, targets, sample_weights)}

  model.fit_generator(
      generator           = train_generator,
      steps_per_epoch     = 28709 //batch_size,  # samples_per_epoch / batch_size
      epochs              = epochs_top_layers, 
      validation_steps    = 3589//batch_size,
      validation_data     = validation_generator)

  # FULL NETWORK TRAINING ###############################################################################################################################

  # At this point, the top layers are well trained and we can start fine-tuning convolutional layers from ResNet-50

  # Fine-tuning of all the layers
  for layer in model.layers:
      layer.trainable = True

  # We need to recompile the model for these modifications to take effect (we use SGD with nesterov momentum and a low learning rate)
      # optimizer:    String (name of optimizer) or optimizer object
          # lr:       float >= 0. Learning rate
          # momentum: float >= 0. Parameter updates momentum
          # decay:    float >= 0. Learning rate decay over each update
          # nesterov: boolean. Whether to apply Nesterov momentum
      # loss:     String (name of objective function) or objective function
      # metrics:  List of metrics to be evaluated by the model during training and testing
  model.compile(
      optimizer   = SGD(lr = 1e-4, momentum = 0.9, decay = 0.0, nesterov = True),
      loss        = 'binary_crossentropy', 
      metrics     = ['accuracy'])

  model.fit_generator(
      generator           = train_generator,
      steps_per_epoch     = 28709 // batch_size,  # samples_per_epoch / batch_size 
      epochs              = epochs_all_layers,                        
      validation_data     = validation_generator,
      validation_steps    = 3589//batch_size,
      callbacks           = [reduce_lr, reduce_lr_plateau, early_stop, checkpointer])