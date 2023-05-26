import os
import sys
import easyocr
import matplotlib.pyplot as plt 

inp_path = sys.argv[1]
out_path = sys.argv[2]

for path, subdirs, files in os.walk(inp_path):
    for name in files:
        
        read_path = os.path.join(path,name)
        save_path = os.path.join(out_path, name)

        print("Reading file: "+read_path)
        print("Saving to: "+save_path)

        reader = easyocr.Reader(['it'], False, recog_network='unicam_model')
        #reader = easyocr.Reader(['it'], False)
        
        result = reader.readtext(read_path)

        im = plt.imread(read_path)

        dpi = 80
        height, width = im.shape
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)

        plt.imshow(im, cmap="gray")

        for _ in result:
            x = [n[0] for n in _[0]]
            y = [n[1] for n in _[0]]
            plt.fill(x,y, facecolor='none', edgecolor='red')
            #plt.text(x[0],y[0], _[1].replace("1", "i" ).replace("0", "o" ), color='red', fontsize=15)
            plt.text(x[0],y[0], _[1], color='red', fontsize=15)

        plt.axis('off')
        plt.savefig(save_path)

