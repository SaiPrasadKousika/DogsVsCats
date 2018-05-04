import numpy as np
import cv2
import os
import random
import tensorflow as tf
import tflearn
from tqdm import tqdm
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import scipy


img_size=64
train_dogs=0
train_cats=0
test_dogs =0 
test_cats =0

def read_train_data():
    filename="generated_train.npy"
    path="MyTrain"
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
    path="MyTest"
    if os.path.exists(filename):
        print ("Loaded existing test data.!")
        return np.load(filename)
    else:
        test_data=[]
        images = os.listdir(path)
        for each in tqdm(images):
            if ('dog' in each):
                label=[0,1]
            elif ('cat' in each):
                label=[1,0]
            img_data = cv2.imread(os.path.join(path,each), cv2.IMREAD_COLOR)
            img_resize = cv2.resize(img_data, (64,64))
            test_data.append([np.array(img_resize),np.array(label)])
        np.save(filename,test_data)
        print ("Generated test data.!")
        return test_data

train_data = read_train_data()
test_data = read_test_data()
np.random.shuffle(train_data)
X = []
Y = []
test_x = []
test_y = []
for each in train_data:
    X.append(each[0])
    Y.append(each[1])
    if (each[1][0]==1):
        train_cats+=1
    else:
        train_dogs+=1

for each in test_data:
    test_x.append(each[0])
    test_y.append(each[1])
    if (each[1][0]==1):
        test_cats+=1
    else:
        test_dogs+=1

X = np.array(X).reshape(-1,img_size,img_size,3)
test_x = np.array(test_x).reshape(-1,img_size,img_size,3)

# In[7]:


#random.shuffle(train_data)
tf.reset_default_graph()
conv_net_1 = input_data(shape=[None, 64, 64,3], name='input')
conv_net_2 = conv_2d(conv_net_1,32,4,activation="relu")
conv_net_3 = max_pool_2d(conv_net_2, 2)

conv_net_4 = conv_2d(conv_net_3,64,4,activation="relu")
conv_net_5 = max_pool_2d(conv_net_4, 2)

conv_net_6 = conv_2d(conv_net_5, 128,4, activation="relu")
conv_net_7 = max_pool_2d(conv_net_6, 2)

conv_net_8 = conv_2d(conv_net_7, 256,4, activation="relu")
conv_net_9 = max_pool_2d(conv_net_8, 2)

conv_net_10 = conv_2d(conv_net_9, 512,4, activation="relu")
conv_net_11 = max_pool_2d(conv_net_10,2)

conv_net_12 = fully_connected(conv_net_11, 1024, activation="relu")
conv_net_13 = dropout(conv_net_12, 0.8)

conv_net_14 = fully_connected(conv_net_13, 2, activation="softmax")
conv_net_final = regression(conv_net_14, optimizer="adam", learning_rate=1e-3, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(conv_net_final,tensorboard_dir='model_log')



model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=0.1, show_metric=True, run_id="DogsVsCats")

correct = 0 
count = 0 
for num,each in tqdm(enumerate(test_x)):
    img=each
    num=test_y[num][0]
    img_resize = img.reshape(-1,img_size,img_size,3)
    predict = model.predict(img_resize)[0]
    if (predict>0.5):
        label=1
    else:
        label=0
    if (num==label):
        correct+=1

if (count!=0):
    print ("Data Statistics \n\nTraining Data: ("+str(train_dogs+train_cats)+")\n\tDogs: "+str(train_dogs)+"\n\tCats: "+str(train_cats))
    print ("\nTesting Data: ("+str(test_cats+test_dogs)+")\n\tDogs: "+str(test_dogs)+"\n\tCats: "+str(test_cats))
    print("\nAccuracy",float(correct)/count)


        

