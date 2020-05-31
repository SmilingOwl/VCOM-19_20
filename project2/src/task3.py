import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers.merge import add, concatenate
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, MaxPooling2D
from keras.models import Model
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC, TruePositives, FalsePositives
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

def obtain_train_generator():
  train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
  )
  batchsize = 64
  train_generator = train_datagen.flow_from_directory(
    directory="drive/My Drive/Colab Notebooks/Data/task3/training",
    classes = ['ISBI2016_ISIC_Part2B_Training_Data'],
    target_size=(256, 256),
    batch_size=batchsize,
    class_mode = None,
    shuffle=True,
    subset='training'
  )

  masks_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
  )
  masks_generator = masks_datagen.flow_from_directory(
    directory="drive/My Drive/Colab Notebooks/Data/task3/training/ISBI2016_ISIC_Part2B_Training_GroundTruth",
    classes = ['globules', 'streaks'],
    target_size=(256, 256),
    batch_size=batchsize,
    class_mode = None,
    shuffle=True,
    subset='training'
  )
  train_generator = zip(train_generator, masks_generator)
  return train_generator

def create_unet_model():
  inputs = Input(shape=(256,256,1))
  x1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
  x1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(x1)
  x2 = MaxPooling2D(pool_size=(2, 2))(x1)
  x2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(x2)
  x2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(x2)
  x3 = MaxPooling2D(pool_size=(2, 2))(x2)
  x3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(x3)
  x3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(x3)
  x4 = MaxPooling2D(pool_size=(2, 2))(x3)
  x4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(x4)
  x4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(x4)
  x5 = MaxPooling2D(pool_size=(2, 2))(x4)
  x5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(x5)
  x5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(x5)
  x6 = UpSampling2D(size = (2, 2))(x5)
  x6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(x6)
  x6 = concatenate([x4, x6])
  x6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(x6)
  x6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(x6)
  x6 = UpSampling2D(size = (2, 2))(x6)
  x6 = Conv2D(256, 2, activation = 'relu', padding = 'same')(x6)
  x6 = concatenate([x3, x6])
  x6 = Conv2D(256, 3, activation = 'relu', padding = 'same')(x6)
  x6 = Conv2D(256, 3, activation = 'relu', padding = 'same')(x6)
  x6 = UpSampling2D(size = (2, 2))(x6)
  x6 = Conv2D(128, 2, activation = 'relu', padding = 'same')(x6)
  x6 = concatenate([x2, x6])
  x6 = Conv2D(128, 3, activation = 'relu', padding = 'same')(x6)
  x6 = Conv2D(128, 3, activation = 'relu', padding = 'same')(x6)
  x6 = UpSampling2D(size = (2, 2))(x6)
  x6 = Conv2D(64, 2, activation = 'relu', padding = 'same')(x6)
  x6 = concatenate([x1, x6])
  x6 = Conv2D(64, 3, activation = 'relu', padding = 'same')(x6)
  x6 = Conv2D(64, 3, activation = 'relu', padding = 'same')(x6)
  x6 = Conv2D(2, 1, padding = 'same')(x6)

  model = Model(inputs, x6)
  metrics = [
    BinaryAccuracy(name='accuracy'),
    Precision(name='precision'),
    Recall(name='recall'),
    AUC(name='auc'),
    TruePositives(name='tp'),
    FalsePositives(name='fp'),
  ]
  model.compile(
      optimizer=Adam(lr=1e-5),
      loss=SparseCategoricalCrossentropy(from_logits=True),
      metrics=metrics
  )
  return model


train_generator = obtain_train_generator()
model = create_unet_model()

checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True, mode='max')
tf.keras.utils.plot_model(model, show_shapes=True)
#model_history = model.fit(train_generator, epochs=20,
#                          steps_per_epoch=30,
#                          validation_steps=VALIDATION_STEPS,
#                          validation_data=test_dataset,
#                          callbacks=[DisplayCallback()])

#TODO: separate train data set in validation set + model.fit + test generator + results

