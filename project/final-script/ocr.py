"""This module uses Easy-OCR library for text-recognition on the images"""

import easyocr

reader = easyocr.Reader(['en'], cudnn_benchmark=True, gpu=True)

def img_to_text(image):
    text = []
    cords = []
    bounds = reader.readtext(image, add_margin=0.2)
    for bound in bounds:
        text.append(bound[1])
        cords.append(bound[2])
    return text

