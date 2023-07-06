"""This module uses Easy-OCR library for text-recognition on the images"""

import easyocr
import os

home_dir = os.path.expanduser("~")
easyocr_dir = home_dir + "\\.EasyOCR"
model_dir = easyocr_dir + "\\model"
user_network_dir = easyocr_dir + "\\user_network"
reader = None


def easy_ocr_standard_model(image):
    global reader
    if reader is None:
        reader = easyocr.Reader(['it'], cudnn_benchmark=True, gpu=True)
    return compute_ocr(image)


def custom_model_machine_written(image):
    global reader
    if reader is None:
        reader = easyocr.Reader(['it'], recog_network='modello_macchina',
                                model_storage_directory="models/",
                                user_network_directory="models/", cudnn_benchmark=True, gpu=False)
    return compute_ocr(image)


def custom_model_hand_written(image):
    global reader
    if reader is None:
        reader = easyocr.Reader(['it'], recog_network='modello_mano',
                                model_storage_directory="models/",
                                user_network_directory="models/", cudnn_benchmark=True, gpu=False)
    return compute_ocr(image)


def compute_ocr(image):
    text = []
    cords = []
    bounds = reader.readtext(image, add_margin=0.2)
    for bound in bounds:
        text.append(bound[1])
        cords.append(bound[2])
    return text


def save_ocr_result(result, path, united):
    file = open(path, "w")
    for s in result:
        file.write(s)
        file.write(" ")
    file.close()
