import numpy as np
from scipy import signal
from PIL import Image
import glob
import cv2

def load_image(path):
    a = cv2.imread(glob.glob(path)[0], 0)
    a2 = np.asarray(a)/255.0
    return a2

def save(path, img):
    tmp = np.asarray(img*255.0, dtype=np.uint8)
    Image.fromarray(tmp).save(path)

def denoise_image(inp):
    # estimate 'background' color by a median filter
    bg = signal.medfilt2d(inp, 21)
    save('background.png', bg)

    # compute 'foreground' mask as anything that is significantly darker than
    # the background
    mask = inp < bg - 0.39
    save('foreground_mask.png', mask)

    # return the input value for all pixels in the mask or pure white otherwise
    return np.where(mask, inp, 1.0)

inp_path = 'test.png'
out_path = 'output.png'

inp = load_image(inp_path)
out = denoise_image(inp)

save(out_path, out)