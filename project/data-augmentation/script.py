import random
import cv2
import numpy as np
import albumentations as A
import os
import sys
from PIL import Image
from augment import augment_img 

inp_path = sys.argv[1]
out_path = sys.argv[2]


f = open(out_path+"labels.csv",'w')
f.write("filename;words\n")
for path, subdirs, files in os.walk(inp_path):
    for name in files:
        filename, extension = os.path.splitext(name)
        read_path = os.path.join(path,name)
        save_path = os.path.join(out_path, str(int(filename)+10000) + '.jpg')
        
        print("Reading file: "+read_path)

        im = Image.open(read_path) 
        augment = augment_img(im)
        augment.save(save_path)
        f.write(str(int(filename)+10000) + '.jpg' + ";\n")