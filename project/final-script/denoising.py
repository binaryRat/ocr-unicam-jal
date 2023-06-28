"""This module uses the Adaptive-Threshold method for noise removal from images"""
import cv2
from skimage.metrics import mean_squared_error,peak_signal_noise_ratio,structural_similarity
import matplotlib.pyplot as plt

matrix_size = 601
threshold = 50


def adaptive_treshold(images):
    den = []
    for img in images:
        temp = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, matrix_size,
                                   threshold)
        den.append(temp)
    return den


def edge_detection(images):
    den = []
    for img in images:
        (H, W) = img.shape[:2]
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        canny = cv2.Canny(blurred, 30, 150)
        img = canny
    return images

