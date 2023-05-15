import easyocr
import matplotlib.pyplot as plt 

reader = easyocr.Reader(['it'], False, recog_network='unicam_model')
#reader = easyocr.Reader(['en'], False)
result = reader.readtext('1694_0000 (601-50).JPG')

im = plt.imread('1694_0000 (601-50).JPG')

dpi = 80
#im_data = plt.imread(im)
height, width = im.shape
figsize = width / float(dpi), height / float(dpi)

fig = plt.figure(figsize=figsize)

plt.imshow(im, cmap="gray")

for _ in result:
    x = [n[0] for n in _[0]]
    y = [n[1] for n in _[0]]
    plt.fill(x,y, facecolor='none', edgecolor='red')
    plt.text(x[0],y[0], _[1], color='red', fontsize=15)

plt.axis('off')
plt.savefig('output_easyocr_unicam.png')