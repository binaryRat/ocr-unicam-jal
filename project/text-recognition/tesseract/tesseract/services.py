import os
import cv2


# Loads all the image in the folder of the given path
def load_img(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
    return images


# Saves an array of img in to the destination path, the images will be saved named
# 1, 2 , 3 .... in the array order
def save_images(images, path):
    count = 1
    for img in images:
        new_path = path + '/' + str(count) + ".JPG"
        count += 1
        cv2.imwrite(new_path, img)


# Resize an img respecting the proportion of the original image
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)
