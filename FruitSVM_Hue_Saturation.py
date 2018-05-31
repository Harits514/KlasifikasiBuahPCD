from __future__ import print_function

import numpy as np
import cv2
import glob
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#Bikin Array Kosong
fruit_images = []
labels = []

#Baca Data
for fruit_dir_path in glob.glob("D:/Fruit-Images-Dataset-master/PCD3/*"):
    fruit_label = fruit_dir_path.split("\\")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path)

        image = cv2.resize(image, (80, 80))

        fruit_images.append(image)
        labels.append(fruit_label)
fruit_images = np.array(fruit_images)
labels = np.array(labels)

#Proses Label
label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
print(id_to_label_dict)

label_ids = np.array([label_to_id_dict[x] for x in labels])

#Hitung Hue dan Saturation Histogram
images_hue_hist=[]
images_sat_hist=[]

for i in fruit_images:
    hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist([hsv],[0],None,[175],[5,180])
    sat_hist = cv2.calcHist([hsv],[1],None,[251],[5,256])
    images_hue_hist.append(hue_hist)
    images_sat_hist.append(sat_hist)
    #hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

new_data = np.concatenate((images_hue_hist,images_sat_hist), axis=1)
print(new_data.shape)

#Split Data Train Test untuk menguji model
X_train, X_test, y_train, y_test = train_test_split(new_data, label_ids, test_size=0.25, random_state=42)
X_tr = np.array(X_train, np.float32)
y_tr = np.array(y_train, np.int32)
X_te = np.array(X_test, np.float32)
y_te = np.array(y_test, np.int32)

#Membuat Model SVM
#Jika ingin membuat model baru Komen bagian Load model
#lalu un-comment dari bagian setup svm sampai save trained model
#Di-save agar bisa dpakai di django dan agar lebih cepat jika
#dijalankan ulang
'''
# Set up SVM for OpenCV 3
svm = cv2.ml.SVM_create()
# Set SVM type
svm.setType(cv2.ml.SVM_C_SVC)

# Train SVM on training data
svm.trainAuto(X_tr, cv2.ml.ROW_SAMPLE, y_tr)

# Save trained model
svm.save("fruit_hs_svm_model2.yml")
'''

# Load trained model
svm = cv2.ml.SVM_load('fruit_hs_svm_model2.yml')

#Predict Test dan hitung akurasi
pred = svm.predict(X_te)[1]

a = []
for i in range(len(pred)):
    a.append(int(pred[i][0]))
mask = a == y_te
correct = np.count_nonzero(mask)
print("svm model accuracy with Hue Saturation Histogram descriptor:", end=" ")
print(correct*100.0/pred.size)

