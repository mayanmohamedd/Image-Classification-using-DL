import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, MultiHeadAttention, Conv2D
from tensorflow.keras.layers import LayerNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import Reading


def vision_transformer(image_size, patch_size, num_classes, d_model, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout_rate=0.1, mlp_dropout=0.1):
    # Image input
    inputs = Input(shape=(image_size, image_size, 1))  # Assuming grayscale images
    
    # Patching images into patches using Conv2D
    x = Conv2D(filters=d_model, kernel_size=(patch_size, patch_size), strides=(patch_size, patch_size))(inputs)
    
    # Positional encoding
    positions = tf.range(start=0, limit=x.shape[1], delta=1)
    positions = tf.expand_dims(positions, 0)
    positions = tf.tile(positions, [tf.shape(x)[0], 1])
    positions = tf.one_hot(positions, depth=image_size // patch_size, dtype=tf.float32)  # Adjusted dimension
    
    # Reshape positions to match the shape of x
    positions = tf.reshape(positions, [-1, tf.shape(x)[1], tf.shape(x)[2], 1])
    
    x = x + positions
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, d_model, num_heads, ff_dim, dropout_rate)
    
    # Classification head
    x = GlobalAveragePooling2D()(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def transformer_block(x, d_model, num_heads, ff_dim, dropout_rate):
    # Multi-head self-attention
    x = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x, x)
    x = Dropout(dropout_rate)(x)
    res = x + LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward layer
    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout_rate)(x)
    x = Dense(d_model)(x)
    x = Dropout(dropout_rate)(x)
    
    return x + res

   


# Example usage
data_dir = "dataset NN'23\\dataset\\train"
num_classes = 5  # Adjust based on your actual number of classes

datagen= ImageDataGenerator(
        rescale=1./255,  # Normalizes pixel values to [0, 1]
        rotation_range=20,  # Randomly rotates images in the range (0-20 degrees)
        width_shift_range=0.1,  # Randomly shifts images horizontally (fraction of total width)
        height_shift_range=0.1,  # Randomly shifts images vertically (fraction of total height)
        shear_range=0.2,  # Shear intensity (angle in counter-clockwise direction)
        zoom_range=0.2,  # Randomly zooms inside images
        horizontal_flip=True,  # Randomly flips images horizontally
        fill_mode='nearest'  # Strategy for filling in newly created pixels
    )

# Load data
data = Reading.create_data(data_dir)
images = np.array([item[0] for item in data]).reshape(-1, 224, 224, 1)  # Assuming grayscale images
labels = np.array([item[1] for item in data])

train_generator = datagen.flow(images, labels, batch_size=128)
augmented_images = train_generator.next() 
datagen.fit(images)
all_X_batches = []
all_Y_batches = []
num_batches = len(train_generator)

for i in range(num_batches):
    batch_X, batch_Y = train_generator.next()  # Get the next batch
    all_X_batches.append(batch_X)  # Collect batches of X
    all_Y_batches.append(batch_Y)  # Collect batches of Y

X_augmented = np.concatenate([images] + all_X_batches, axis=0)

# Concatenate all Y batches along axis=0 (samples axis)
Y_augmented = np.concatenate([labels] + all_Y_batches, axis=0)

# Train the model
model = vision_transformer(image_size=224, patch_size=16, num_classes=num_classes,
                           d_model=256, num_heads=8, ff_dim=1024, num_transformer_blocks=12, mlp_units=[2048, 1024])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.fit(X_augmented, Y_augmented, epochs=1   , batch_size=32, validation_split=0.2)