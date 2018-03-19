import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from cyvlfeat.sift.dsift import dsift
from sklearn.svm import LinearSVC, SVC
#from cyvlfeat.hog import hog
from sklearn.externals import joblib
from skimage.feature import hog

pei = np.load('pei.npy')
mis = np.load('mis.npy')
#Xin = np.load('Xin.npy')
hand8 = np.load('hand8.npy')
hand31 = np.load('hand31.npy')
hand48 = np.load('hand48.npy')
hand61 = np.load('hand61.npy')
hand70 = np.load('hand70.npy')
hand67 = np.load('hand67.npy')
#hand2 = np.load('hand2.npy')
#hand3 = np.load('hand3.npy')
label = [1,1,1,1,1,1,1,1,
2,2,2,2,2,2,2,2,
3,3,3,3,3,3,3,3,
4,4,4,4,4,4,4,4,
5,5,5,5,5,5,5,5,
6,6,6,6,6,6,6,6,
7,7,7,7,7,7,7,7,
8,8,8,8,8,8,8,8,
9,9,9,9,9,9,9,9,
10,10,10,10,10,10,10,10]
#label = [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10]
train = np.zeros((80, 2304), dtype='float32')

for num in range(10):
  #pei[num] = (pei[num] - np.mean(pei[num]))/np.std(pei[num])
  #mis[num] = (mis[num] - np.mean(mis[num]))/np.std(mis[num])
  train[8*num,:] = np.reshape(pei[num], (1, 2304))
  #train[9*num+1,:] = np.reshape(Xin[num], (1, 22500))
  train[8*num+1,:] = np.reshape(mis[num], (1, 2304))
  train[8*num+2,:] = np.reshape(hand8[num], (1, 2304))
  train[8*num+3,:] = np.reshape(hand31[num], (1, 2304))
  train[8*num+4,:] = np.reshape(hand48[num], (1, 2304))
  train[8*num+5,:] = np.reshape(hand61[num], (1, 2304))
  train[8*num+6,:] = np.reshape(hand70[num], (1, 2304))
  train[8*num+7,:] = np.reshape(hand67[num], (1, 2304))

clf = LinearSVC()#SVC(kernel = 'rbf', random_state=0, gamma = 0.2, C=10) ##LinearSVC()#SVC(kernel = 'rbf', random_state=0, gamma = 0.2, C=10) #
clf.fit(train, label)

# Save the classifier
#joblib.dump(clf, "digits_cls_comp_hogsk_svm.pkl", compress=3)

for x in range(1,11):
  filename = '/home/henry/DSP/data/xin/%d.png'%x
  img = imread(filename)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  resize = cv2.resize(gray, (100,100))
  #descriptors = hog(resize, 2)
  descriptors = hog(resize, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(1, 1), visualise=False)
  test_image_feats = np.reshape(descriptors, (1, 2304))
  pred_label = clf.predict(test_image_feats)
  print pred_label

