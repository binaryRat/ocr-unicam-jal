"""This module uses the Adaptive-Threshold method for noise removal from images"""
import cv2

matrix_size = 601
threshold = 50


def adaptive_treshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, matrix_size,
                                 threshold)


def edge_detection(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(blurred, 30, 150)
    return canny
