import os
import cv2


def load_images(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


def save_images(images, path):
    count = 1
    for img in images:
        new_path = path + '/' + str(count) + ".jpg"
        count += 1
        cv2.imwrite(new_path, img)


def save_image(image, path):
    cv2.imwrite(path, image)


def get_filename(filename):
    name = os.path.splitext(filename)
    return name[0]


def concatenate_filename(filename, string):
    old = os.path.splitext(filename)
    new = str(old[0]) + string + str(old[1])
    return new
