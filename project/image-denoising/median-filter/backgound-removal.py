import numpy as np
import sys
from scipy import signal
from PIL import Image
import glob
import cv2

def load_image(path):
    a = cv2.imread(glob.glob(path)[0], 0)
    #a2 = np.asarray(a)/255.0
    return a

def save(path, img):
    tmp = np.asarray(img, dtype=np.uint8)
    Image.fromarray(tmp).save(path)

def denoise_image(inp):
    # Stima del 'background' utilizzando Median filtering
    bg = signal.medfilt2d(inp, 20001)
    save('background.png', bg)

    # Computazione della maschera di 'foreground'come tutto ciò che risulta piò scuro del background 
    print(inp)
    mask = inp < (bg - 50)
    
    save('foreground_mask.png', mask)

    # Ritorno dei pixel dell'immagine originale sotratta la maschera
    return np.where(mask, inp, 1.0)

inp_path = sys.argv[1]
out_path = sys.argv[2]

inp = load_image(inp_path)
out = denoise_image(inp)

save(out_path, out)