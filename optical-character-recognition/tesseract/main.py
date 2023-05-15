import services
from PIL import Image
import pytesseract
import cv2

hand_written = services.load_img('input/mano')
machine_written = services.load_img('input/macchina')

for img in hand_written:
    print()




