import numpy as np
from random import shuffle
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
import keras
import tflearn
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, ZeroPadding2D
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression




class Reading_Data:
    def create_label(class_number):
        """ Create a one-hot encoded vector from the numerical class label """
        num_classes = 5  # Update this if you have a different number of classes
        label = np.zeros(num_classes)
        label[class_number - 1] = 1
        return label
    
    def Data_augmentation(img):
        datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

        return datagen

    def create_train_data(data_dir):
        data = []
        #class_path: dataset NN'23\dataset\train\1,2,...,5
        for class_folder in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_folder)

            if os.path.isdir(class_path):
                class_number = int(class_folder)

                for img in (os.listdir(class_path)):
                    path = os.path.join(class_path, img)
                    img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img_data = cv2.resize(img_data, (50, 50))
                    # Normalize pixel values
                    normalized_image = img_data / 255.0
                    #aug_img = Reading_Data.Data_augmentation(normalized_image)
                    label = Reading_Data.create_label(class_number)
                    data.append([np.array(normalized_image), label])

            shuffle(data)
        return data
    
    def create_test_data(data_dir):
        data = []

        for img_file in os.listdir(data_dir):
            path = os.path.join(data_dir, img_file)
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (50, 50))
            # Normalize pixel values
            normalized_image = img_data / 255.0
            aug_img = Reading_Data.Data_augmentation(normalized_image)
            data.append(np.array(aug_img))

        return data



class AlexNet:
    
    def init_AlexNet():
        # block1
        IMG_SIZE = 50 
        LR = 0.001
        conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
        conv1_0 = conv_2d(conv_input, 96, 11, strides=4, activation='linear')  # Using 'linear' activation before BatchNormalization
        Norm1 = BatchNormalization()(conv1_0)
        conv1_0afteractv = Activation('relu')(Norm1)  # Applying activation after BatchNormalization
        pool1 = max_pool_2d(conv1_0afteractv, 3, strides=2)
        
        #block 2
        conv2_0 = conv_2d(pool1, 256, 5)
        Norm2 = BatchNormalization()(conv2_0)
        pad2 = ZeroPadding2D(padding=(2, 2))(Norm2)
        conv2_0afteractv=Activation('relu')(pad2)
        pool2 = max_pool_2d(conv2_0afteractv, 3, strides=2)
        
        
        #block 3
        conv3_0 = conv_2d(pool2, 384, 3)
        Norm3 = BatchNormalization()(conv3_0)
        pad3 = ZeroPadding2D(padding=(1, 1))(Norm3)
        conv3_0afteractv=Activation('relu')(pad3)
        
        #block 3.1
        
        conv4_0 = conv_2d(conv3_0afteractv, 384, 3)
        Norm4 = BatchNormalization()(conv4_0)
        pad4 = ZeroPadding2D(padding=(1, 1))(Norm4)
        conv4_0afteractv=Activation('relu')(pad4)
        
        
        #block 3.2
        conv5_0 = conv_2d(conv4_0afteractv, 256, 3)
        Norm5 = BatchNormalization()(conv5_0)
        pad5 = ZeroPadding2D(padding=(1, 1))(Norm5)
        conv5_1=Activation('relu')(pad5)
        pool3 = max_pool_2d(conv5_1, 3, strides=2)
        
        
        #block 4
        fully_layer = fully_connected(pool3, 4096, activation='relu')
        fully_layer1 = fully_connected(fully_layer, 4096, activation='relu')
        Alexnet_layers = fully_connected(fully_layer1, 5, activation='softmax')
        Alexnet_layers = regression(Alexnet_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
        
        model = tflearn.DNN(Alexnet_layers, tensorboard_dir='log', tensorboard_verbose=3)
        return model

    
    

    def model_fit(model,x,y,MODEL_NAME):
        model.fit({'input': x}, {'targets': y}, n_epoch=10, snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    
#=====================================================================
import tflearn
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD
from keras.losses import CategoricalCrossentropy
import numpy as np
import cv2
import os

IMG_SIZE = 50
NUM_CLASSES = 5

def create_label(class_number):
    label = np.zeros(NUM_CLASSES)
    label[class_number - 1] = 1
    return label

def create_data(data_dir):
    data = []
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)

        if os.path.isdir(class_path):
            class_number = int(class_folder)

            for img in os.listdir(class_path):
                path = os.path.join(class_path, img)
                img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
                img_data = img_data / 255.0  # Normalization
                label = create_label(class_number)
                data.append([np.array(img_data), label])

    np.random.shuffle(data)
    return data

def create_test_data(TEST_DIR):
    testing_data=[]
    for img in (os.listdir(TEST_DIR)):
        img_name, img_extension = os.path.splitext(img)  # Split name and extension
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), img_name])  # Use img_name without extension
    np.random.shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

def predict_test_data(test_data):
    X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    predictions = model.predict(X_test)
    # 'predictions' will contain the predictions made by the model for the test data
    return predictions

train = create_data("C:\\Users\\PC\\Downloads\\dataset NN'23\\dataset\\train")
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array([i[1] for i in train])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

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




model.fit(X_train, y_train, epochs=72, validation_data=(X_val, y_val))


TEST_DIR = ("C:\\Users\\PC\\Downloads\\dataset NN'23\\dataset\\test")
test_data = create_test_data(TEST_DIR)
predictions = predict_test_data(test_data)

import csv

# Writing predictions to a CSV file
with open('predictions_sc_59.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image_id', 'label'])  # Writing headers
    for i in range(len(test_data)):
        writer.writerow([test_data[i][1], predictions[i].argmax()])  # Writing image name and corresponding prediction