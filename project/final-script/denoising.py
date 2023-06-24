"""This module uses the Adaptive-Threshold method for noise removal from images"""
import cv2

matrix_size = 601
threshold = 50


def denoise(images):
    den = []
    for img in images:
        temp = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, matrix_size,
                                   threshold)
        den.append(temp)
    return den

