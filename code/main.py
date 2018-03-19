# Import the modules
import cv2
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
#from cyvlfeat.hog import hog
from skimage.io import imread
from skimage import io
import scipy.misc
from function_ver2 import Find_Char
import argparse
from skimage.feature import hog

#Which char user change
parser = argparse.ArgumentParser()
parser.add_argument('--char',default='pei', help='Choose which character you want to transforming?')
args = parser.parse_args()

# Load the classifier
clf = joblib.load("digits_cls_comp_hogsk_svm.pkl")

# Read the input image 
file_name = "/home/henry/new_hog/find_char_graph_15.png"#test.PNG#find_char_graph_15.png"
img_o = cv2.imread(file_name)
char, loc = Find_Char(file_name,35,3,False)
print img_o.shape

#testing
final = np.zeros([img_o.shape[0],img_o.shape[1],3] ,dtype=np.uint8)
final.fill(255) 
for num in range(len(char)):
   char_r = cv2.resize(char[num], (100,100))
   #print char_r.shape
   #roi_hog_fd = hog(char_r, 2)
   roi_hog_fd = hog(char_r, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualise=False)
   result = clf.predict(np.reshape(roi_hog_fd, (1, 22500)))
   filen = '/home/henry/DSP/data/%s/%d.png'%(args.char,result)
   img = cv2.imread(filen)
   resize = cv2.resize(img, (int(char[num].shape[1]*0.8),int(char[num].shape[0]*0.8)))
   final[loc[num][0]:int(loc[num][0]+char[num].shape[0]*0.8), loc[num][1]:int(loc[num][1]+char[num].shape[1]*0.8), :] = resize

plt.figure(2)
plt.imshow(final)
plt.show()

