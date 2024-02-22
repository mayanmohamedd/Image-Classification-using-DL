import warnings

import tflearn
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import to_categorical
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
#from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
import numpy as np
# import cv2
# import os
import matplotlib.pyplot as plt
import tensorflow as tf
# from sklearn.svm import SVC
import Reading

IMG_SIZE = 50
NUM_CLASSES = 5





# def predict_test_data(test_data):
#     X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#     predictions = model.predict(X_test)
#     # 'predictions' will contain the predictions made by the model for the test data
#     return predictions

train = Reading.create_Alex_data("dataset NN'23\\dataset\\train", IMG_SIZE)
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array([i[1] for i in train])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizes pixel values to [0, 1]
    rotation_range=20,  # Randomly rotates images in the range (0-20 degrees)
    width_shift_range=0.1,  # Randomly shifts images horizontally (fraction of total width)
    height_shift_range=0.1,  # Randomly shifts images vertically (fraction of total height)
    shear_range=0.2,  # Shear intensity (angle in counter-clockwise direction)
    zoom_range=0.2,  # Randomly zooms inside images
    horizontal_flip=True,  # Randomly flips images horizontally
    fill_mode='nearest'  # Strategy for filling in newly created pixels
)

train_generator = datagen.flow(X_train, y_train, batch_size=128)

# Generate some augmented images
augmented_images = train_generator.next()  # Retrieves the next batch of augmented images

# Plot the first few augmented images
num_images_to_plot = 5
fig, axes = plt.subplots(1, num_images_to_plot, figsize=(15, 5))

for i in range(num_images_to_plot):
    axes[i].imshow(augmented_images[0][i])  # Show the ith image from the batch
    axes[i].axis('off')  # Hide axis labels

plt.tight_layout()
plt.show()

datagen.fit(X_train)

# Concatenate the batches with X_train and Y_train
all_X_batches = []
all_Y_batches = []

num_batches = len(train_generator)  # Calculate the number of batches

for i in range(num_batches):
    batch_X, batch_Y = train_generator.next()  # Get the next batch
    all_X_batches.append(batch_X)  # Collect batches of X
    all_Y_batches.append(batch_Y)  # Collect batches of Y

# Concatenate all X batches along axis=0 (samples axis)
X_augmented = np.concatenate([X_train] + all_X_batches, axis=0)

# Concatenate all Y batches along axis=0 (samples axis)
Y_augmented = np.concatenate([y_train] + all_Y_batches, axis=0)




def AlexNet():
    inp = Input((IMG_SIZE, IMG_SIZE, 1))
    x = Conv2D(96, 11, strides=4, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D(3, 2)(x)
    x = Conv2D(256, 5, strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(3, 2)(x)
    x = Conv2D(384, 3, strides=1, activation='relu',padding='same')(x)
    x = Conv2D(384, 3, strides=1, activation='relu',padding='same')(x)
    x = Conv2D(256, 3, strides=1, activation='relu',padding='same')(x)
   # x = MaxPooling2D(5, 2)(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    return model

model = AlexNet()
optimizer = SGD(learning_rate=0.001, momentum=0.9)

model.compile(loss=CategoricalCrossentropy(),
              optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=70, validation_data=(X_val, y_val))


# TEST_DIR = ("C:\\Users\\PC\\Downloads\\dataset NN'23\\dataset\\test")
# test_data = Reading.create_test_data(TEST_DIR,IMG_SIZE)
# predictions = predict_test_data(test_data)

# import csv

# # Writing predictions to a CSV file
# with open('predictions_sc_59.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Image_id', 'label'])  # Writing headers
#     for i in range(len(test_data)):
#         writer.writerow([test_data[i][1], predictions[i].argmax()+ 1])  # Writing image name and corresponding prediction