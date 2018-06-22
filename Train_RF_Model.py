from _ImageTools import *
from _ProjectTools import *

'''
Program utama untuk membuat model untuk klasifikasi 5 buah dengan menggunakan
algoritma random forest. Program memerlukan file _ImageTools.py dan _ProjectTools.py
berada pada directory yang sama.

Library yang perlu diinstall: pywt, numpy, opencv-python(cv2), matplotlib, glob, os,
time, random, sklearn dan cPickle
'''

#Mencatat Waktu awal
start_time = time.time()

#Data Train
fruit_images, y_train, id_to_label_dict = read_image("D:/Fruit-Images-Dataset-master/PCD3/*")#Pastikan di belakang nama directory terdapat /*

print(fruit_images.shape)
print(id_to_label_dict)

x_train = ExtractFeature(fruit_images)

#Data Test
fruit_test, y_test, id_to_label_dict = read_image("D:/Fruit-Images-Dataset-master/Validation/*")#Pastikan di belakang nama directory terdapat /*
x_test = ExtractFeature(fruit_test)

#Train, Test, Save, and accuracy test
model = RandomForestClassifier(n_estimators=12,criterion='gini',
                               max_features='auto')

model = model.fit(x_train, y_train)
save_model(model, "_model-data.pkl")#Pastikan string berformat .pkl

test_predict = model.predict(x_test)

print_accuracy(y_test,test_predict,start_time)
print_importance(model)
