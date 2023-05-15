from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt

import os
import cv2
from keras.utils.image_utils import img_to_array

from tqdm import tqdm  # progress bar for iterable

from sklearn.model_selection import train_test_split

np.random.seed(42)

# x is noisy data and y is clean data
SIZE = 520

# noisy data retrieving
noisy_data = []
path1 = 'dataset/train'
files = os.listdir(path1)
for i in tqdm(files):
    img = cv2.imread(path1 + '/' + i, cv2.IMREAD_UNCHANGED)  # Change 0 to 1 for color
    img = cv2.resize(img, (SIZE, SIZE))
    # converts the img in an 3D array and adds in to the noisy data
    noisy_data.append(img_to_array(img))

# cleaned data retrieving for training
clean_data = []
path2 = 'dataset/train_cleaned'
files = os.listdir(path2)
for i in tqdm(files):
    img = cv2.imread(path2 + '/' + i, cv2.IMREAD_UNCHANGED)  # Change 0 to 1 for color images
    img = cv2.resize(img, (SIZE, SIZE))
    # converts the img in an 3D array and adds in to the noisy data
    clean_data.append(img_to_array(img))

# changes the array dimension
noisy_train = np.reshape(noisy_data, (len(noisy_data), SIZE, SIZE, 1))
noisy_train = noisy_train.astype('float32') / 255.

clean_train = np.reshape(clean_data, (len(clean_data), SIZE, SIZE, 1))
clean_train = clean_train.astype('float32') / 255.

# Displaying images with noise
plt.figure(figsize=(10, 2))  # creates a new figure
for i in range(1, 5):
    ax = plt.subplot(1, 4, i)
    plt.imshow(noisy_train[i].reshape(SIZE, SIZE), cmap="gray")  # change to 'binary' for black and white
plt.show()

# Displaying clean images
plt.figure(figsize=(10, 2))
for i in range(1, 5):
    ax = plt.subplot(1, 4, i)
    plt.imshow(clean_train[i].reshape(SIZE, SIZE), cmap="gray")
plt.show()

# compression -> decompression sequence
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.summary()

x_train, x_test, y_train, y_test = train_test_split(noisy_train, clean_train, test_size=0.20, random_state=0)

model.fit(x_train, y_train, epochs=10, batch_size=8, shuffle=True, verbose=1, validation_split=0.1)

print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(x_test), np.array(y_test))[1] * 100))

model.save('denoising_autoencoder.model')

no_noise_img = model.predict(x_test)

plt.imshow(no_noise_img[i].reshape(SIZE, SIZE), cmap="gray")

# shows 10 images in the original version and the denoised version
plt.figure(figsize=(40, 4))
for i in range(10):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(y_test[i].reshape(SIZE, SIZE), cmap="binary")
    # display reconstructed (after noise removed) image
    ax = plt.subplot(3, 20, 40 + i + 1)
    plt.imshow(no_noise_img[i].reshape(SIZE, SIZE), cmap="gray")
plt.show()
