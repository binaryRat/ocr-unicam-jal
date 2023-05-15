import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import math
import cv2
from sewar.full_ref import uqi, psnr, rmse, ssim
import pandas as pd
sns.set()
from scipy import signal
import glob

def adaptive_thresholding(noisy_img):
    adaptive_th=cv2.adaptiveThreshold(noisy_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,301,23)
    return adaptive_th

def printImage(image, title, filename):
        font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
        plt.rcParams["axes.grid"] = False
        fig=plt.figure(figsize=(12,8))
        plt.imshow(image, cmap='gray')
        plt.title(title, fontdict=font)
        plt.savefig(filename)

#img_paths=os.listdir('train_cleaned')
#img_paths=['train_cleaned/'+x for x in img_paths]
#cleaned_img=[cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in img_paths]

#img_paths=os.listdir('train')
#img_paths=['train/'+x for x in img_paths]
#dirty_img=[cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in img_paths]

#denoised=cleaned_img[4]
#noisy=dirty_img[4]

def load_image(path):
    a = cv2.imread(glob.glob(path)[0], cv2.IMREAD_GRAYSCALE)
    #a2 = np.asarray(a)/255.0
    return a

def save(path, img):
    tmp = np.asarray(img, dtype=np.uint8)
    Image.fromarray(tmp).save(path)

inp_path = sys.argv[1]
out_path = sys.argv[2]

print("Reading file :"+inp_path)
inp = load_image(inp_path)
adaptive_th=adaptive_thresholding(inp)

#save(out_path, adaptive_th)

#printImage(adaptive_th, 'Resultant Image after adaprive', "adaptive.png")

save(out_path, adaptive_th)


