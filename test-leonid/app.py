import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
import cv2
from sewar.full_ref import uqi, psnr, rmse, ssim
import pandas as pd
sns.set()
from scipy import signal
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
#%matplotlib inline

def train(cleaned_img, dirty_img):
        
        regressor = LinearRegression()

        for i in range(len(cleaned_img)):
                clean=cleaned_img[i]
                dirty=dirty_img[i]

                dirty_flat=dirty.flatten()
                x=np.reshape(dirty_flat, (dirty.shape[0]*dirty.shape[1], 1))
                clean_flat=clean.flatten()
                y=np.reshape(clean_flat, (clean.shape[0]*clean.shape[1], 1))

                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=10)
                  
                regressor.fit(X_train, y_train)
                print("Trained image :"+str(i))
                print(str(regressor.coef_))

        dirty=dirty_img[0]

        printImage(dirty, 'Dirty Image', "dirty.png")

        dirty_flat=dirty.flatten()
        x=np.reshape(dirty_flat, (dirty.shape[0]*dirty.shape[1], 1))
        result=regressor.predict(x)
        result=np.reshape(result,(dirty.shape[0], dirty.shape[1]))
        
        #print(rmse(clean, result)) #RMSE
        #print(uqi(clean, result)) #UQI
        #print(psnr(clean, result)) #PSNR
        printImage(result, 'Resultant Image after Linear Regression', "result.png")

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

img_paths=os.listdir('train_cleaned')
img_paths=['train_cleaned/'+x for x in img_paths]
img_paths.index('train_cleaned/72.png')
cleaned_img=[cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in img_paths]

img_paths=os.listdir('train')
img_paths=['train/'+x for x in img_paths]
dirty_img=[cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in img_paths]


train(cleaned_img, dirty_img)



