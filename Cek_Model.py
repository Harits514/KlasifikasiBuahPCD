from __future__ import print_function

import numpy as np
import cv2
from matplotlib import pyplot as plt

#Catatan label : {0: 'Apple Golden', 1: 'Apple Granny Smith', 2: 'Apple Red', 3: 'Lemon', 4: 'Mandarine'}

#Test satu gambar untuk melihat hasil
img = cv2.imread("gambar/gold2.jpg")
img=cv2.resize(img, (80,80))
cv2.imshow("Orisinil",img)
test1 = []

test1.append(img)

images_hue_hist=[]
images_sat_hist=[]

for i in test1:
    hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist([hsv],[0],None,[175],[5,180])
    sat_hist = cv2.calcHist([hsv],[1],None,[251],[5,256])
    images_hue_hist.append(hue_hist)
    images_sat_hist.append(sat_hist)
    plt.plot(hue_hist)
    plt.plot(sat_hist)

new_data = np.concatenate((images_hue_hist,images_sat_hist), axis=1)
print(new_data.shape)
X_tr = np.array(new_data, np.float32)
# print(X_tr)

svm = cv2.ml.SVM_load('fruit_hs_svm_model2.yml')

pred=svm.predict(X_tr)[1]

print(pred[0][0])

plt.show()

cv2.waitKey(0)
