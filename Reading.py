import numpy as np
import os
import cv2
from sklearn.utils import shuffle

def create_label(class_number, num_classes=5):
    """ Create a one-hot encoded vector from the numerical class label """
    label = np.zeros(num_classes)
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
                img_data = cv2.resize(img_data, (224, 224))
                label = create_label(class_number)

                data.append([np.array(img_data), label])

    # Shuffle the data
    data = shuffle(data)
    return data




def create_Alex_data(data_dir,IMG_SIZE):
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

# def create_test_data(TEST_DIR,IMG_SIZE):
#     testing_data=[]
#     for img in (os.listdir(TEST_DIR)):
#         img_name, img_extension = os.path.splitext(img)  # Split name and extension
#         path = os.path.join(TEST_DIR, img)
#         img_data = cv2.imread(path, 0)
#         img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
#         testing_data.append([np.array(img_data), img_name])  # Use img_name without extension
#     np.random.shuffle(testing_data)
#     np.save('test_data.npy', testing_data)
#     return testing_data