"""This module uses Easy-OCR library for text-recognition on the images"""

import easyocr
import os
import shutil

home_dir = os.path.expanduser("~")
easyocr_dir = home_dir + "\\.EasyOCR"
model_dir = easyocr_dir + "\\model"
user_network_dir = easyocr_dir + "\\user_network"
shutil.copy('models/modello_macchina.pth', model_dir)
shutil.copy('models/modello_macchina.py', user_network_dir)
shutil.copy('models/modello_macchina.yaml', user_network_dir)

reader = easyocr.Reader(['en'], recog_network='modello_macchina', cudnn_benchmark=True, gpu=True)


def img_to_text(image):
    text = []
    cords = []
    bounds = reader.readtext(image, add_margin=0.2)
    for bound in bounds:
        text.append(bound[1])
        cords.append(bound[2])
    return text
