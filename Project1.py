import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# Load images
train_dir = '.'
validation_dir = '.'

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_dataset = train_datagen.flow_from_directory(train_dir,
                                                 target_size=(200,200),
                                                 batch_size=3,
                                                 class_mode='binary')

validation_dataset = validation_datagen.flow_from_directory(validation_dir,
                                                           target_size=(200,200),
                                                           batch_size=3,
                                                           class_mode='binary')

# Create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['acc'])

# Train the model
history = model.fit(train_dataset,
                    steps_per_epoch=3,
                    epochs=30,
                    validation_data=validation_dataset)