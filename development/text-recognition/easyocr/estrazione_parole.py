import os
import sys
import cv2
import easyocr
import matplotlib.pyplot as plt 
import numpy as np


reader = easyocr.Reader(['it'], False) # gpu

inp_path = sys.argv[1]
out_path = sys.argv[2]
data = {}
f = open(out_path+"labels.csv",'w')
f.write("filename;words\n")

i=1273
for path, subdirs, files in os.walk(inp_path):
    for name in files:
        read_path = os.path.join(path,name)

        print("Reading file: "+read_path)

        #reader = easyocr.Reader(['it'], False)
        im = cv2.imread(read_path)
        result = reader.readtext(im)

        dpi = 80
        height, width, test = im.shape
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)

        #plt.imshow(im, cmap="gray")

        list_confidence = []

        for detection in result:
            top_left = tuple([int(val) for val in detection[0][0]])
            bottom_right = tuple([int(val) for val in detection[0][2]])
            cropped_img = im[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :] 
            dpi = 80
            height, width, test = cropped_img.shape
            figsize = width / float(dpi), height / float(dpi)

            if(height != 0 and width != 0):
                fig = plt.figure(figsize=figsize)
                plt.imsave(out_path + str(i) + '.jpg', cropped_img)
                plt.close()
                f.write(str(i) + '.jpg' + ";" + detection[1] + "\n")
                i = i + 1

f.close()
