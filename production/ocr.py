"""This module uses Easy-OCR library for text-recognition on the images"""

import easyocr
import os
import shutil

home_dir = os.path.expanduser("~")
easyocr_dir = home_dir + "\\.EasyOCR"
model_dir = easyocr_dir + "\\model"
user_network_dir = easyocr_dir + "\\user_network"


def easy_ocr_standard_model(image):
    reader = easyocr.Reader(['it'], cudnn_benchmark=True, gpu=True)
    return compute_ocr(reader, image)


def custom_model_machine_written(image):
    #shutil.copy('models/machine-written-model/modello_macchina.pth', model_dir)
    #shutil.copy('models/machine-written-model/modello_macchina.py', user_network_dir)
    #shutil.copy('models/machine-written-model/modello_macchina.yaml', user_network_dir)
    reader = easyocr.Reader(['it'], recog_network='modello_macchina', model_storage_directory="models/machine-written-model/", user_network_directory="models/machine-written-model/",cudnn_benchmark=True, gpu=True)
    return compute_ocr(reader, image)


def custom_model_hand_written(image):
    #shutil.copy('models/hand-written-model/modello_macchina.pth', model_dir)
    #shutil.copy('models/hand-written-model/modello_macchina.py', user_network_dir)
    #shutil.copy('models/hand-written-model/modello_macchina.yaml', user_network_dir)
    reader = easyocr.Reader(['it'], recog_network='modello_mano', model_storage_directory="models/hand-written-model/", user_network_directory="models/hand-written-model/", cudnn_benchmark=True, gpu=True)
    return compute_ocr(reader, image)


def compute_ocr(reader, image):
    text = []
    cords = []
    bounds = reader.readtext(image, add_margin=0.2)
    for bound in bounds:
        text.append(bound[1])
        cords.append(bound[2])
    return text
