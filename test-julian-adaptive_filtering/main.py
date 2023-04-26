import numpy as np
from skimage.io import imread
from skimage import color, data, restoration
from scipy.signal import convolve2d
from skimage.util import random_noise
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio


def display(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


# uploading image
test_img = imread("input/0.png")
print("Type: ", type(test_img))
print("dtype: ", test_img.dtype)
print("shape: ", test_img.shape)
display(test_img)

# displaying sample image
img = color.rgb2gray(data.astronaut())
display(img)

# degradation of the image
k = 5
psf = np.ones((k, k)) / (k * k)
print(psf)

# blurring the image
img1 = convolve2d(img, psf, 'same')
display(img1)

# adding noise to image
img1 = random_noise(img1, mode='gaussian')
display(img1)

# restoring the image with unsupervised wiener
img_restored, chains = restoration.unsupervised_wiener(img1, psf=psf)
display(img_restored)

# restoring with wiener
img_restored2 = restoration.wiener(img1, psf=psf, balance=0.35)
display(img_restored2)

# noise_ratio_original = peak_signal_noise_ratio(image_true=test_img, image_test=img1)
# noise_ratio_method1 = peak_signal_noise_ratio(image_true=img, image_test=img_restored)
# noise_factor_method2 = peak_signal_noise_ratio(image_true=img, image_test=img_restored2)
#
# print(f"Peak Signal Noise Ratio blur image: {noise_ratio_original:.3f}")
# print(f"Peak Signal Noise Ratio method 1: {noise_ratio_method1:.3f}")
# print(f"Peak Signal Noise Ratio method 2: {noise_factor_method2:.3f}")
