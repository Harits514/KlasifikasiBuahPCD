import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import cPickle

def save_model(model, string):
    with open(string, 'wb') as save:
        cPickle.dump(model, save)

def print_accuracy(y_test,test_predict,start_time):
    precision = accuracy_score(y_test,test_predict) * 100
    print("Model Accuracy: {0:.3f}".format(precision))
    print("--- %s seconds ---\n" % (time.time() - start_time))

def print_importance(model):
    feat_names = ['Red mean', 'Green mean', 'Blue mean',
              'Red std', 'Green std', 'Blue std',
              'Value mean', 'Saturation mean', 'Hue mean',
              'Value std', 'Saturation std', 'Hue std',
              'Energy', 'Entropy']
    importances = model.feature_importances_
    indices = np.argsort(importances)
    for name,importance in zip(feat_names,importances):
        print(name+'='+str(importance))

    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)))
    plt.xlabel('Relative Importance')
    plt.show()
