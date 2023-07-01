import numpy as np
import os
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

matrix_size = 601
threshold = 50
def adaptive_thresholding(noisy_img):
    adaptive_th=cv2.adaptiveThreshold(noisy_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,matrix_size,threshold)
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

def load_image(path):
    a = cv2.imread(glob.glob(path)[0], cv2.IMREAD_GRAYSCALE)
    #a2 = np.asarray(a)/255.0
    return a

def save(path, img):
    tmp = np.asarray(img, dtype=np.uint8)
    Image.fromarray(tmp).save(path)

inp_path = sys.argv[1]
out_path = sys.argv[2]

for path, subdirs, files in os.walk(inp_path):
    for name in files:
        
        read_path = os.path.join(path,name)
        save_name = os.path.splitext(name)[0]+" ("+str(matrix_size)+"-"+str(threshold)+")"+os.path.splitext(name)[1]
        
        save_path = os.path.join(out_path, save_name)
        print("Reading file: "+read_path)
        print("Saving to: "+save_path)
        inp = load_image(read_path)
        adaptive_th=adaptive_thresholding(inp)
        
        save(save_path, adaptive_th)




