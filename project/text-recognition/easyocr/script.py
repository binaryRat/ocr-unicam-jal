import os
import sys
import cv2
import easyocr
import matplotlib.pyplot as plt 
import numpy as np


#reader = easyocr.Reader(['en'], False, recog_network='unicam_project',cudnn_benchmark=True)
#reader = easyocr.Reader(['it'], recog_network='modello_macchina',cudnn_benchmark=True, gpu=True) # False
reader = easyocr.Reader(['it'], recog_network='modello_macchina_aug', gpu=True) # False
inp_path = sys.argv[1]
out_path = sys.argv[2]
data = {}

for path, subdirs, files in os.walk(inp_path):
    for name in files:
        
        read_path = os.path.join(path,name)
        save_path = os.path.join(out_path, name)

        print("Reading file: "+read_path)

        #reader = easyocr.Reader(['it'], False)
        im = cv2.imread(read_path)
        result = reader.readtext(im, width_ths=0.3)
        #result = reader.recognize(im)
        #print(result)

        dpi = 80
        height, width, test = im.shape
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)

        plt.imshow(im, cmap="gray")

        list_confidence = []

        for _ in result:
            x = [n[0] for n in _[0]]
            y = [n[1] for n in _[0]]
            list_confidence.append(round(_[2],3))
            if _[2]>=0.9:
                plt.fill(x,y, facecolor='none', edgecolor='green')
                plt.text(x[0],y[0], _[1] + " "+str(round(_[2],2)), color='green', fontsize=15)
            else:
                plt.fill(x,y, facecolor='none', edgecolor='red')
                plt.text(x[0],y[0], _[1], color='red', fontsize=15)

        data[name] = list_confidence

        plt.axis('off')
        plt.savefig(save_path)
        plt.close()

f = open(out_path+"risultati.csv",'w')
f.write("file,avg,median,75th,90th,95th\n")

for file in data.keys():
    num = np.array(data[file])
    f.write(
        file  + "," +
        str(round(np.mean(num),3)) + "," +
        str(round(np.percentile(num,50),3)) + "," +
        str(round(np.percentile(num,75),3)) + "," +
        str(round(np.percentile(num,90),3)) + "," +
        str(round(np.percentile(num,95),3))
    )
    f.write("\n")
f.close()