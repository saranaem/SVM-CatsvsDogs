import glob
import numpy as np
from PIL import Image
import pandas as pd
from skimage.feature import hog
from skimage.transform import resize
from sklearn import svm
import cv2


train = []
test=[]

#dogs>>0  cats>>1
target_train=[0]*1000
target_train.extend([1]*1000)

target_test=[0]*100
target_test.extend([1]*100)

target_test=pd.DataFrame(target_test)
target_train=pd.DataFrame(target_train)

count=0
for filename in glob.glob('train/dog*.jpg'):
    im=cv2.imread(filename)
    im = resize(im, (128, 64))
    train.append(im)
    count+=1
    if(count==1100):
        break

test=train[:100]
train=train[100:]

count=0
for filename in glob.glob('train/cat*.jpg'):
    im=cv2.imread(filename)
    im = resize(im, (128, 64))
    train.append(im)
    count+=1
    if(count==1100):
        break

test.extend(train[2000:])
train=train[:2000]

#feature extraction for the train
fd = []
for img in train:
    fdt, hog_image=hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
    fd.append(fdt)
target_train =  np.array(target_train).reshape(len(target_train),1)
df_train = np.hstack((fd,target_train))
np.random.shuffle(df_train)
#feature extraction for the test
fd_test=[]
for img in test:
    fdt, hog_image=hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
    fd_test.append(fdt)

target_test =  np.array(target_test).reshape(len(target_test),1)
df_test = np.hstack((fd_test,target_test))
np.random.shuffle(df_test)
#fitting
clf = svm.SVC(kernel='poly', degree=3).fit(df_train[:,:-1],df_train[:,-1])
predictions = clf.predict(df_test[:,:-1])
accuracy_test = np.mean(predictions == df_test[:,-1])
print("test accuracy")
print(accuracy_test)

predictions_train = clf.predict(df_train[:, :-1])
accuracy_train = np.mean(predictions_train == df_train[:, -1])
print("train accuracy")
print(accuracy_train)
