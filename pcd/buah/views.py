# -*- coding: utf-8 -*-
from django.shortcuts import render

from django.http import HttpResponseRedirect, HttpResponse
from django.template import loader
from django.shortcuts import get_object_or_404, render, redirect
from django.core.validators import RegexValidator
from django.contrib import messages

from django.http import Http404
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.core import serializers
from django.conf import settings
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

import os

from .models import *

class home(generic.ListView):
	model=fruit
	template_name = 'buah/home.html'

class hasil(generic.ListView):
	model=fruit
	template_name = 'buah/hasil.html'
	context_object_name = 'aja'

	def get_queryset(self):
		ay = self.kwargs['oy']
		a = fruit.objects.get(id=ay)
		print(a.Tipe)
		print(a.id)
		print(a.document)
		return fruit.objects.get(id=ay)

def upload_pic(request):
    if request.POST:
        file_doc = request.FILES['ahoy']
        sv = fruit(document = file_doc)
        sv.save()
        oyo = fruit.objects.last()
        oy=oyo.id

        item=fruit.objects.get(id=oy)
        #item = fruit.objects.get(id=oy)
        img_file = item.document
        ayay=img_file.url
        print(ayay)
        print(img_file)
        image = cv2.imread(img_file.url)
        img=cv2.resize(image, (80,80))
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
        new_data = np.concatenate((images_hue_hist,images_sat_hist), axis=1)
        X_tr = np.array(new_data, np.float32)
        svm = cv2.ml.SVM_load('fruit_hs_svm_model2.yml')
        pred=svm.predict(X_tr)[1]
        fr=pred[0][0]
        fra=int(fr)
        item.Tipe=fra
        item.save()

    return HttpResponseRedirect(reverse('buah:hasil', args=[item.id]))

# #image = cv2.imdecode(numpy.fromstring(img_file, numpy.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
# SZ=20
# bin_n = 16 # Number of bins
# winSize = (60,60)
# blockSize = (30,30)
# blockStride = (10,10)
# cellSize = (10,10)
# nbins = 9
# derivAperture = 1
# winSigma = -1.
# histogramNormType = 0
# L2HysThreshold = 0.2
# gammaCorrection = 1
# nlevels = 64
#
# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#
# img=cv2.resize(image, (60,60))
# test1 = []
# test1.append(img)
# images_hogged=([hog.compute(i) for i in test1])
