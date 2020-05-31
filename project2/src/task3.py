import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers.merge import add, concatenate
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_batchsize = 64
val_batchsize = 134

train_generator = train_datagen.flow_from_directory(
        directory="../Data/Train",
        target_size=(224, 224),
        batch_size=train_batchsize,
        class_mode='binary', 
        shuffle=True,
        seed=42)

validation_generator = validation_datagen.flow_from_directory(
        directory="/Data/Valid",
        target_size=(224, 224),
        batch_size=val_batchsize,
        class_mode='binary',
        shuffle=False, 
        seed=42)


def create_unet_model():
    inputs = Input(shape=(None,None,1))
    x1 = Conv2D(64, 3, activation = 'relu', padding = 'valid')(inputs)
    x1 = Conv2D(64, 3, activation = 'relu', padding = 'valid')(x1)
    x2 = MaxPooling2D(pool_size=(2, 2))(x1)
    x2 = Conv2D(128, 3, activation = 'relu', padding = 'valid')(x2)
    x2 = Conv2D(128, 3, activation = 'relu', padding = 'valid')(x2)
    x3 = MaxPooling2D(pool_size=(2, 2))(x2)
    x3 = Conv2D(256, 3, activation = 'relu', padding = 'valid')(x3)
    x3 = Conv2D(256, 3, activation = 'relu', padding = 'valid')(x3)
    x4 = MaxPooling2D(pool_size=(2, 2))(x3)
    x4 = Conv2D(512, 3, activation = 'relu', padding = 'valid')(x4)
    x4 = Conv2D(512, 3, activation = 'relu', padding = 'valid')(x4)
    x5 = MaxPooling2D(pool_size=(2, 2))(x4)
    x5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid')(x5)
    x5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid')(x5)
    x6 = UpSampling2D(2)(x5)
    x6 = Conv2D(512, 2, activation = 'relu', padding = 'valid')(x6)
    x6 = concatenate([x4, x6])
    x6 = Conv2D(512, 3, activation = 'relu', padding = 'valid')(x6)
    x6 = Conv2D(512, 3, activation = 'relu', padding = 'valid')(x6)
    x6 = concatenate([x3, x6])
    x6 = Conv2D(256, 3, activation = 'relu', padding = 'valid')(x6)
    x6 = Conv2D(256, 3, activation = 'relu', padding = 'valid')(x6)
    x6 = concatenate([x2, x6])
    x6 = Conv2D(128, 3, activation = 'relu', padding = 'valid')(x6)
    x6 = Conv2D(128, 3, activation = 'relu', padding = 'valid')(x6)
    x6 = concatenate([x1, x6])
    x6 = Conv2D(64, 3, activation = 'relu', padding = 'valid')(x6)
    x6 = Conv2D(64, 3, activation = 'relu', padding = 'valid')(x6)
    x6 = Conv2D(2, 1, activation = 'relu', padding = 'valid')(x6)

    return Model(inputs, x6) 