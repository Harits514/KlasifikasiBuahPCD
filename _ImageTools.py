import pywt
import numpy as np
import cv2
import glob
import os
import random
from sklearn.metrics.cluster import entropy

def read_image(string, x):
    fruit_images = []
    labels = []
    for fruit_dir_path in glob.glob(string):
        images = []
        fruit_label = fruit_dir_path.split("\\")[-1]
        #print(fruit_label)
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            image = cv2.resize(image, (70, 70))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            images.append(image)
        images = np.array(images)
        for i in random.sample(range(0,images.shape[0]),x):
            image = images[i]
            fruit_images.append(image)
            labels.append(fruit_label)
    fruit_images = np.array(fruit_images)
    labels = np.array(labels)

    label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
    id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

    label_ids = np.array([label_to_id_dict[x] for x in labels])

    return(fruit_images, label_ids, id_to_label_dict)

def ExtractFeature(fruit_images):
    list_of_vectors = []
    for img in fruit_images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row,col = gray_img.shape
        canvas = np.zeros((row, col, 1), np.uint8)
        for i in range(row):
            for j in range(col):
                if gray_img[i][j] < 220:
                    canvas.itemset((i, j, 0), 255)
                else:
                    canvas.itemset((i, j, 0), 0)

        kernel = np.ones((3,3),np.uint8)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)

        for i in range(row):
            for j in range(col):
                b,g,r = img[i][j]
                if canvas[i][j] == 255:
                    img.itemset((i, j, 0), b)
                    img.itemset((i, j, 1), g)
                    img.itemset((i, j, 2), r)
                else:
                    img.itemset((i, j, 0), 0)
                    img.itemset((i, j, 1), 0)
                    img.itemset((i, j, 2), 0)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rgb_means, rgb_std = cv2.meanStdDev(img)
        hsv_means, hsv_std = cv2.meanStdDev(hsv_img)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coeff = pywt.dwt2(gray_img, "haar")  # ---- Dekomposisi lv 1
        LL, (LH, HL, HH) = coeff
        Energy = (LH**2 + HL**2 + HH**2).sum()/img.size
        Entropy = entropy(gray_img)

        b, g, r = img[row/2-1,col/2-1]
        # center_code = tools.CentreClass(b,g,r)
        list_of_vectors.append([rgb_means[2], rgb_means[1], rgb_means[0],
                                rgb_std[2], rgb_std[1], rgb_std[0],
                                hsv_means[2], hsv_means[1], hsv_means[0],
                                hsv_std[2], hsv_std[1], hsv_std[0],
                                Energy, Entropy])

    list_of_vectors = np.array(list_of_vectors)
    return (list_of_vectors)
