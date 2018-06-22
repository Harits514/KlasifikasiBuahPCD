from __future__ import print_function

from sklearn.externals import joblib
from _ImageTools import *
from _ProjectTools import *

#{0: 'Apple Golden', 1: 'Apple Granny Smith', 2: 'Apple Red', 3: 'Lemon', 4: 'Mandarine'}
img = cv2.imread("gambar/mandarine2.jpg")
img = cv2.medianBlur(img,5)
fruit_images = []
fruit_images.append(img)

X_tr = ExtractFeature(fruit_images)

clf = joblib.load('clfRF70-12-g-wcc.pkl')
test_predict = clf.predict(X_tr)

print("Kelas: {0: 'Apple Golden', 1: 'Apple Granny Smith', 2: 'Apple Red', 3: 'Lemon', 4: 'Mandarine'}")
print("Kelas hasil prediksi:",end="")
print(int(test_predict))

print(test_predict[0])

cv2.waitKey(0)
