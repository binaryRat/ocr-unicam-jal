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


def save_ocr_result(results, path, unified):
    if unified:
        unified_path = path + "/unified.txt"
        unified_file = open(unified_path, "w")
    counter = 1
    for result in results:
        name = path + "/" + str(counter) + ".txt"
        file = open(name, "w")
        for s in result:
            file.write(s)
            file.write(" ")
            if unified:
                unified_file.write(s)
                unified_file.write(" ")
        file.close()
        if unified:
            unified_file.write("\n------------------------------------\n")
        counter += 1
