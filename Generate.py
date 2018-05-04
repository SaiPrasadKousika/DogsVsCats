import numpy as np
import cv2
import os
import random
import tensorflow as tf
import tflearn
from tqdm import tqdm
import tflearn
from PIL import Image 
from PIL import ImageFilter
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import scipy
img_size=64


# In[2]:


def read_train_data():
    filename="generated_train.npy"
    path="train"
    if os.path.exists(filename):
        print ("Loaded existing train data.!")
        return np.load(filename)
    else:
        train_data=[]
        images = os.listdir(path)
        for each in tqdm(images):
            if ('dog' in each):
                label=[0,1]
            elif ('cat' in each):
                label=[1,0]
            img_data = cv2.imread(os.path.join(path,each), cv2.IMREAD_COLOR)
            img_resize = cv2.resize(img_data, (64,64))
            train_data.append([np.array(img_resize),np.array(label)])
        np.save(filename,train_data)
        print ("Generated train data.!")
        random.shuffle(train_data)
        return train_data
def read_test_data():
    filename="generated_test.npy"
    path="test"
    if os.path.exists(filename):
        print ("Loaded existing test data.!")
        return np.load(filename)
    else:
        test_data=[]
        images = os.listdir(path)
        for each in tqdm(images):
            name =each.split(".")[0]
            img_data = cv2.imread(os.path.join(path,each), cv2.IMREAD_COLOR)
            img_resize = cv2.resize(img_data, (64,64))
            test_data.append([np.array(img_resize),name])
        np.save(filename,test_data)
        print ("Generated test data.!")
        return test_data

train_data = read_train_data()
test_data = read_test_data()
random.shuffle(train_data)


np.random.shuffle(train_data)
train_split=train_data[:-5000]
test_split=train_data[-5000:]
X = []
Y = []
test_x = []
test_y = []
for each in train_split:
    X.append(each[0])
    Y.append(each[1])
for each in test_split:
    test_x.append(each[0])
    test_y.append(each[1])

## Train Data

dogs=0
cats=0
for num,img in tqdm(enumerate(X)):
    if Y[num][0]==1: 
        dogs+=1
        scipy.misc.imsave("MyTrain/dog."+str(dogs)+'.jpg', img)
    else:
        cats+=1
        scipy.misc.imsave("MyTrain/cat."+str(cats)+'.jpg', img)

## Validataion Data

dogs=0
cats=0
for num,img in tqdm(enumerate(test_x)):
    if test_y[num][0]==1: 
        dogs+=1
        scipy.misc.imsave("MyTest/dog."+str(dogs)+'.jpg', img)
    else:
        cats+=1
        scipy.misc.imsave("MyTest/cat."+str(cats)+'.jpg', img)


## Test Data

for each in tqdm(test_data):
    
    img = each[0]
    scipy.misc.imsave("OrgTest/"+str(each[1])+'.jpg', img)
    
    