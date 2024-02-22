# import cv2
# import numpy as np
# import os
# from PIL import Image
# from collections import Counter
# # from random import shuffle
# import tflearn
# import csv
# import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.core import input_data, fully_connected
# from tflearn.layers.estimator import regression
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.models import Sequential
# from sklearn.model_selection import train_test_split
# import Reading

# K.clear_session()

# IMG_SIZE = 50
# LR = 0.001
# MODEL_NAME = 'vgg 16-cnn'




# # Function to convert images to JPEG format
# def convert_to_jpeg(image_path):
#     try:
#         img = Image.open(image_path)

#         # Convert image to RGB mode if it's in palette mode (mode 'P')
#         if img.mode == 'P':
#             img = img.convert('RGB')

#         # Convert image to RGB mode if it has an alpha channel (mode 'RGBA')
#         if img.mode == 'RGBA':
#             img = img.convert('RGB')

#         new_path = os.path.splitext(image_path)[0] + ".jpeg"
#         img.save(new_path, "JPEG", quality=95)  # Save as JPEG with 95% quality
#         return True
#     except Exception as e:
#         print(f"Error converting {image_path}: {e}")
#         return False

# train_dir = "dataset NN'23\\dataset\\train"

# # Collect file extensions in the directory
# file_extensions = []

# for class_folder in os.listdir(train_dir):
#     class_path = os.path.join(train_dir, class_folder)

#     if os.path.isdir(class_path):
#         for img_file in os.listdir(class_path):
#             _, extension = os.path.splitext(img_file.lower())  # Get file extension
#             file_extensions.append(extension)

# # Count occurrences of each extension
# extension_counts = Counter(file_extensions)

# # Find the most common extension
# most_common_extension = extension_counts.most_common(1)
# print("Most common extension:", most_common_extension)


# # Convert images in the directory to JPEG
# for root, _, files in os.walk(train_dir):
#     for file in files:
#         if file.lower().endswith(('.png')):
#             file_path = os.path.join(root, file)
#             # if convert_to_jpeg(file_path):
#                 # print(f"Converted {file_path} to JPEG")


# train = Reading.create_data("dataset NN'23\\dataset\\train")
# # print(len(train[0][0]))

# x_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
# y_train = [i[1] for i in train]

# X_train, X_test, Y_train,Y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=10, shuffle=True)

# Y_train = np.array(Y_train)
# Y_test = np.array(Y_test)

# print("x_test",X_test.shape)

# model = Sequential()

# # block1
# conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
# conv1_0 = conv_2d(conv_input, 64, 3, activation='relu')
# conv1_1 = conv_2d(conv1_0, 64, 3, activation='relu')
# pool1 = max_pool_2d(conv1_1, 2, strides=2)

# # block 2
# conv2_0 = conv_2d(pool1, 128, 3, activation='relu')
# conv2_1 = conv_2d(conv2_0, 128, 3, activation='relu')
# pool2 = max_pool_2d(conv2_1, 2, strides=2)

# # block 3
# conv3_0 = conv_2d(pool2, 256, 3, activation='relu')
# conv3_1 = conv_2d(conv3_0, 256, 3, activation='relu')
# conv3_2 = conv_2d(conv3_1, 256, 3, activation='relu')
# pool3 = max_pool_2d(conv3_2, 2, strides=2)

# # block 4
# conv4_0 = conv_2d(pool3, 512, 3, activation='relu')
# conv4_1 = conv_2d(conv4_0, 512, 3, activation='relu')
# conv4_2 = conv_2d(conv4_1, 512, 3, activation='relu')
# pool4 = max_pool_2d(conv4_2, 2, strides=2)

# # block 5
# conv5_0 = conv_2d(pool4, 512, 3, activation='relu')
# conv5_1 = conv_2d(conv5_0, 512, 3, activation='relu')
# conv5_2 = conv_2d(conv5_1, 512, 3, activation='relu')
# pool5 = max_pool_2d(conv5_2, 2, strides=2)


# fully_layer = fully_connected(pool5, 4096, activation='relu')
# fully_layer1 = fully_connected(fully_layer, 4096, activation='relu')
# VGG_layers = fully_connected(fully_layer1, 5, activation='softmax')

# VGG_layers = regression(VGG_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
# model = tflearn.DNN(VGG_layers, tensorboard_dir='log', tensorboard_verbose=3)

# datagen = ImageDataGenerator(
#     rescale=1./255,  # Normalizes pixel values to [0, 1]
#     rotation_range=20,  # Randomly rotates images in the range (0-20 degrees)
#     width_shift_range=0.1,  # Randomly shifts images horizontally (fraction of total width)
#     height_shift_range=0.1,  # Randomly shifts images vertically (fraction of total height)
#     shear_range=0.2,  # Shear intensity (angle in counter-clockwise direction)
#     zoom_range=0.2,  # Randomly zooms inside images
#     horizontal_flip=True,  # Randomly flips images horizontally
#     fill_mode='nearest'  # Strategy for filling in newly created pixels
# )

# train_generator = datagen.flow(X_train, Y_train, batch_size=128)

# augmented_images = train_generator.next()  # Retrieves the next batch of augmented images

# num_images_to_plot = 5
# fig, axes = plt.subplots(1, num_images_to_plot, figsize=(15, 5))

# for i in range(num_images_to_plot):
#     axes[i].imshow(augmented_images[0][i])
#     axes[i].axis('off')

# plt.tight_layout()
# plt.show()

# datagen.fit(x_train)

# # Concatenate the batches with X_train and Y_train
# all_X_batches = []
# all_Y_batches = []

# num_batches = len(train_generator)

# for i in range(num_batches):
#     batch_X, batch_Y = train_generator.next()
#     all_X_batches.append(batch_X)
#     all_Y_batches.append(batch_Y)

# # Concatenate all X batches along axis=0 (samples axis)
# X_augmented = np.concatenate([X_train] + all_X_batches, axis=0)

# # Concatenate all Y batches along axis=0 (samples axis)
# Y_augmented = np.concatenate([Y_train] + all_Y_batches, axis=0)

# print(X_augmented.shape)
# print(len(Y_augmented))

# if os.path.exists('model1.tfl.meta'):
#     model.load('./model1.tfl')

# else:
#    model.fit({'input': X_augmented}, {'targets': Y_augmented}, n_epoch=20 ,validation_set=({'input': X_test}, {'targets': Y_test}),snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
#    model.save('model1.tfl')


# # test_folder_path = "dataset NN'23\\dataset\\train"

# # with open('predictions.csv', mode='w', newline='') as file:
# #     writer = csv.writer(file)
# #     writer.writerow(['image_id', 'label'])

# #     # Iterate through the test images
# #     for filename in os.listdir(test_folder_path):
# #         # Construct the full path to the image
# #         img_path = os.path.join(test_folder_path, filename)

# #         # Load and preprocess the test image
# #         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# #         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# #         test_img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)  # Add batch dimension
# #         test_img = test_img / 255.0  # Normalize pixel values

# #         # Make predictions
# #         predictions = model.predict(test_img)[0]

# #         # Get the predicted label (index of the maximum value in predictions)
# #         predicted_label = np.argmax(predictions)


# #         # Write image ID (filename without extension) and predicted label to CSV
# #         writer.writerow([os.path.splitext(filename)[0], predicted_label + 1])  

# #         # Display the image
# #         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# #         plt.axis('off')





import cv2
import numpy as np
import os
from PIL import Image
from collections import Counter
from random import shuffle
import tflearn
import csv
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split


K.clear_session()

IMG_SIZE = 50
LR = 0.001
MODEL_NAME = 'vgg 16-cnn'


def create_label(class_number):
    """ Create a one-hot encoded vector from the numerical class label """
    num_classes = 5  # Update this if you have a different number of classes
    label = np.zeros(num_classes)
    label[class_number - 1] = 1
    return label


def create_data(data_dir):
    data = []
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)

        if os.path.isdir(class_path):
            class_number = int(class_folder)

            for img in (os.listdir(class_path)):
                path = os.path.join(class_path, img)
                img_data = cv2.imread(path)
                img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                img_data = cv2.resize(img_data, (50, 50))
                label = create_label(class_number)
                data.append([np.array(img_data), label])

        shuffle(data)
    return data

# Function to convert images to JPEG format
def convert_to_jpeg(image_path):
    try:
        img = Image.open(image_path)

        # Convert image to RGB mode if it's in palette mode (mode 'P')
        if img.mode == 'P':
            img = img.convert('RGB')

        # Convert image to RGB mode if it has an alpha channel (mode 'RGBA')
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        new_path = os.path.splitext(image_path)[0] + ".jpeg"
        img.save(new_path, "JPEG", quality=95)  # Save as JPEG with 95% quality
        return True
    except Exception as e:
        print(f"Error converting {image_path}: {e}")
        return False

train_dir = "dataset NN'23\\dataset\\train"

# Collect file extensions in the directory
file_extensions = []

for class_folder in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_folder)

    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            _, extension = os.path.splitext(img_file.lower())  # Get file extension
            file_extensions.append(extension)

# Count occurrences of each extension
extension_counts = Counter(file_extensions)

# Find the most common extension
most_common_extension = extension_counts.most_common(1)
print("Most common extension:", most_common_extension)


# Convert images in the directory to JPEG
for root, _, files in os.walk(train_dir):
    for file in files:
        if file.lower().endswith(('.png')):
            file_path = os.path.join(root, file)
            # if convert_to_jpeg(file_path):
                # print(f"Converted {file_path} to JPEG")


train = create_data("dataset NN'23\\dataset\\train")

x_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train = [i[1] for i in train]

X_train, X_test, Y_train,Y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=10, shuffle=True)

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

print("x_test",X_test.shape)

model = Sequential()

# block1
conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
conv1_0 = conv_2d(conv_input, 64, 3, activation='relu')
conv1_1 = conv_2d(conv1_0, 64, 3, activation='relu')
pool1 = max_pool_2d(conv1_1, 2, strides=2)

# block 2
conv2_0 = conv_2d(pool1, 128, 3, activation='relu')
conv2_1 = conv_2d(conv2_0, 128, 3, activation='relu')
pool2 = max_pool_2d(conv2_1, 2, strides=2)

# block 3
conv3_0 = conv_2d(pool2, 256, 3, activation='relu')
conv3_1 = conv_2d(conv3_0, 256, 3, activation='relu')
conv3_2 = conv_2d(conv3_1, 256, 3, activation='relu')
pool3 = max_pool_2d(conv3_2, 2, strides=2)

# block 4
conv4_0 = conv_2d(pool3, 512, 3, activation='relu')
conv4_1 = conv_2d(conv4_0, 512, 3, activation='relu')
conv4_2 = conv_2d(conv4_1, 512, 3, activation='relu')
pool4 = max_pool_2d(conv4_2, 2, strides=2)

# block 5
conv5_0 = conv_2d(pool4, 512, 3, activation='relu')
conv5_1 = conv_2d(conv5_0, 512, 3, activation='relu')
conv5_2 = conv_2d(conv5_1, 512, 3, activation='relu')
pool5 = max_pool_2d(conv5_2, 2, strides=2)


fully_layer = fully_connected(pool5, 4096, activation='relu')
fully_layer1 = fully_connected(fully_layer, 4096, activation='relu')
VGG_layers = fully_connected(fully_layer1, 5, activation='softmax')

VGG_layers = regression(VGG_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(VGG_layers, tensorboard_dir='log', tensorboard_verbose=3)

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

train_generator = datagen.flow(X_train, Y_train, batch_size=128)

augmented_images = train_generator.next()  # Retrieves the next batch of augmented images

num_images_to_plot = 5
fig, axes = plt.subplots(1, num_images_to_plot, figsize=(15, 5))

for i in range(num_images_to_plot):
    axes[i].imshow(augmented_images[0][i])
    axes[i].axis('off')

plt.tight_layout()
plt.show()

datagen.fit(x_train)

# Concatenate the batches with X_train and Y_train
all_X_batches = []
all_Y_batches = []

num_batches = len(train_generator)

for i in range(num_batches):
    batch_X, batch_Y = train_generator.next()
    all_X_batches.append(batch_X)
    all_Y_batches.append(batch_Y)

# Concatenate all X batches along axis=0 (samples axis)
X_augmented = np.concatenate([X_train] + all_X_batches, axis=0)

# Concatenate all Y batches along axis=0 (samples axis)
Y_augmented = np.concatenate([Y_train] + all_Y_batches, axis=0)

print(X_augmented.shape)
print(len(Y_augmented))

if os.path.exists('model1.tfl.meta'):
    model.load('./model1.tfl')

else:
   model.fit({'input': X_augmented}, {'targets': Y_augmented}, n_epoch=20 ,validation_set=({'input': X_test}, {'targets': Y_test}),snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
   model.save('model1.tfl')


test_folder_path = "dataset NN'23\\dataset\\test"

with open('predictions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_id', 'label'])

    # Iterate through the test images
    for filename in os.listdir(test_folder_path):
        # Construct the full path to the image
        img_path = os.path.join(test_folder_path, filename)

        # Load and preprocess the test image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        test_img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)  # Add batch dimension
        test_img = test_img / 255.0  # Normalize pixel values

        # Make predictions
        predictions = model.predict(test_img)[0]

        # Get the predicted label (index of the maximum value in predictions)
        predicted_label = np.argmax(predictions)


        # Write image ID (filename without extension) and predicted label to CSV
        writer.writerow([os.path.splitext(filename)[0], predicted_label + 1])

        # Display the image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')