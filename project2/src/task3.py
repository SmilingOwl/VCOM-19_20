import random as rn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers.merge import add, concatenate
from keras.layers import Conv2D, Conv2DTranspose, Dropout, Input, UpSampling2D, MaxPooling2D, Concatenate
from keras.models import Model
from keras.metrics import Accuracy, BinaryAccuracy, Precision, Recall, AUC, TruePositives, FalsePositives, MeanIoU, TrueNegatives, FalseNegatives
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt

def obtain_train_generator():
  train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
  )
  batchsize = 80
  train_generator = train_datagen.flow_from_directory(
    directory="/content/Dataset/training",
    classes = ['ISBI2016_ISIC_Part2B_Training_Data'],
    target_size=(128, 128),
    batch_size=batchsize,
    color_mode="rgb",
    class_mode = None,
    shuffle=True
  )

  masks_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
  )
  masks_generator = masks_datagen.flow_from_directory(
    directory="/content/Dataset/training/ISBI2016_ISIC_Part2B_Training_GroundTruth",
    classes = ['globules'],#['globules', 'streaks'],
    target_size=(128, 128),
    batch_size=batchsize,
    color_mode="rgb",
    class_mode = None
  )
  train_generator = zip(train_generator, masks_generator)
  for (img, mask) in train_generator:
    yield (img, mask)

def obtain_validation_generator():
  valid_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
  )
  batchsize = 1
  valid_generator = valid_datagen.flow_from_directory(
    directory="/content/Dataset/validation",
    classes = ['data'],
    target_size=(128, 128),
    batch_size=batchsize,
    color_mode="rgb",
    class_mode = None,
    shuffle=False
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
    directory="/content/Dataset/validation/groundtruth",
    classes = ['globules'],#['globules', 'streaks'],
    target_size=(128, 128),
    batch_size=batchsize,
    color_mode="rgb",
    class_mode = None
  )
  valid_generator = zip(valid_generator, masks_generator)
  for (img, mask) in valid_generator:
    yield (img, mask)

def obtain_test_generator():
  test_datagen = ImageDataGenerator(
    rescale=1./255
  )
  batchsize = 1
  test_generator = test_datagen.flow_from_directory(
    directory="/content/Dataset/test",
    classes = ['ISBI2016_ISIC_Part2B_Test_Data'],
    target_size=(128, 128),
    batch_size=batchsize,
    color_mode="rgb",
    class_mode = None,
    shuffle=False
  )

  masks_datagen = ImageDataGenerator(
    rescale=1./255
  )
  masks_generator = masks_datagen.flow_from_directory(
    directory="/content/Dataset/test/ISBI2016_ISIC_Part2B_Test_GroundTruth",
    classes = ['globules'],#['globules', 'streaks'],
    target_size=(128, 128),
    batch_size=batchsize,
    color_mode="rgb",
    class_mode = None,
    shuffle=False
  )
  test_generator = zip(test_generator, masks_generator)
  for (img, mask) in test_generator:
    yield (img, mask)

def obtain_test_generator_without_masks():
  test_datagen = ImageDataGenerator(
    rescale=1./255
  )
  batchsize = 1
  test_generator = test_datagen.flow_from_directory(
    directory="/content/Dataset/test",
    classes = ['ISBI2016_ISIC_Part2B_Test_Data'],
    target_size=(128, 128),
    batch_size=batchsize,
    color_mode="rgb",
    class_mode = None,
    shuffle=False
  )
  return test_generator

def dice_loss(y_true, y_pred):
  return 1-dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
  return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def jaccard_loss(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
  sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
  jac = (intersection + smooth) / (sum_ - intersection + smooth)
  return (1 - jac) * smooth

def create_unet_model():
  inputs = Input(shape=(128, 128, 3))
  x1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(inputs)
  x1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(x1)
  x2 = MaxPooling2D(pool_size=(2, 2))(x1)
  x2 = Dropout(0.25)(x2)
  x2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(x2)
  x2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(x2)
  x3 = MaxPooling2D(pool_size=(2, 2))(x2)
  x3 = Dropout(0.5)(x3)
  x3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(x3)
  x3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(x3)
  x4 = MaxPooling2D(pool_size=(2, 2))(x3)
  x4 = Dropout(0.5)(x4)
  x4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(x4)
  x4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(x4)
  x5 = MaxPooling2D(pool_size=(2, 2))(x4)
  x5 = Dropout(0.5)(x5)
  x5 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same')(x5)
  x5 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same')(x5)
  x6 = Conv2DTranspose(512, (3, 3), strides = (2, 2), padding = 'same')(x5)
  x6 = Concatenate()([x4, x6])
  x6 = Dropout(0.5)(x6)
  x6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(x6)
  x6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(x6)
  x6 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(x6)
  x6 = Concatenate()([x3, x6])
  x6 = Dropout(0.5)(x6)
  x6 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(x6)
  x6 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(x6)
  x6 = Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same')(x6)
  x6 = Concatenate()([x2, x6])
  x6 = Dropout(0.5)(x6)
  x6 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(x6)
  x6 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(x6)
  x6 = Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same')(x6)
  x6 = Concatenate()([x1, x6])
  x6 = Dropout(0.5)(x6)
  x6 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(x6)
  x6 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(x6)
  x6 = Conv2D(3, (1, 1), activation = 'sigmoid', padding = 'same')(x6)

  model = Model(inputs, x6)

  metrics = [
    BinaryAccuracy(name='accuracy'),
    Precision(name='precision'),
    Recall(name='recall'),
    TruePositives(name='tp'),
    FalsePositives(name='fp'),
    TrueNegatives(name='tn'),
    FalseNegatives(name='fn'),
    MeanIoU(name='iou', num_classes=2),
    dice_coef
  ]
  
  model.compile(
      optimizer=Adam(lr=3e-5),
      loss='binary_crossentropy',
      metrics=metrics
  )
  #tf.keras.utils.plot_model(model, show_shapes=True)
  #model.summary()
  
  return model

model = create_unet_model()

EPOCHS = 5
checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
train_generator = obtain_train_generator()
validation_generator = obtain_validation_generator()
model_history = model.fit_generator(
    train_generator, epochs=EPOCHS,
    steps_per_epoch=8,
    validation_steps=167,
    validation_data=validation_generator,
    callbacks=[checkpoint],
    verbose=1,
    use_multiprocessing=False,
    max_queue_size=10,   
    workers=1
)

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

test_generator = obtain_test_generator()
#results = model.predict_generator(obtain_test_generator_without_masks(), steps=335, verbose=1)
results = model.evaluate(test_generator, steps=335)
print("Loss: " + str(results[0]))
print("Accuracy: " + str(results[1]))
print("Precision: " + str(results[2]))
print("Recall: " + str(results[3]))
print("TP: " + str(results[4]))
print("FP: " + str(results[5]))

