"""This module uses the Adaptive-Threshold method for noise removal from images"""
import cv2

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
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        canny = cv2.Canny(blurred, 30, 150)
        den.append(canny)
    return den
