from PIL import Image
import pytesseract
from pytesseract import Output
import cv2

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

"""
OCR Engine mode
0 Legacy engine only
1 Neural nets LSTM engine only
2 Legacy + LSTM engines
3 Default, based on what is available 

Page Segmentation mode 
0 Orientation and script detection (OSD) only.
1 Automatic page segmentation with OSD.
2 Automatic page segmentation, but no OSD, or OCR. (not implemented)
3 Fully automatic page segmentation, but no OSD. (Default)
4 Assume a single column of text of variable sizes.
5 Assume a single uniform block of vertically aligned text.
6 Assume a single uniform block of text.
7 Treat the image as a single text line.
8 Treat the image as a single word.
9 Treat the image as a single word in a circle.
10 Treat the image as a single character.
11 Sparse text. Find as much text as possible in no particular order.
12 Sparse text with OSD.
13 Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
"""

custom_config = r'--oem 3 --psm 3'


def get_img_info(img):
    return pytesseract.image_to_osd(img)


def img_to_string(img):
    return pytesseract.image_to_string(img, config=custom_config, lang='ita')


def img_to_img_by_char(img, destination_path):
    height, width, channel = img.shape
    boxes = pytesseract.image_to_boxes(img, config=custom_config)
    for box in boxes.splitlines():
        box = box.split(" ")
        img = cv2.rectangle(img, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), (0, 255, 0),
                            2)


def img_to_img_by_words(img, destination_path):
    data = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT, lang='ita')
    amount_boxes = len(data['text'])
    for i in range(amount_boxes):
        if float(data['conf'][i]) > 1:  # change this value for the confidence
            (x, y, width, height) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            img = cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
            img = cv2.putText(img, data['text'][i], (x, y + height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                              cv2.LINE_AA)
    cv2.imwrite(destination_path, img)


def img_to_pdf(img, destination_path):
    pdf = pytesseract.image_to_pdf_or_hocr(r'input\macchina\1.JPG', config=custom_config, extension='pdf', lang='ita')
    with open(destination_path, 'w+b') as f:
        f.write(pdf)
