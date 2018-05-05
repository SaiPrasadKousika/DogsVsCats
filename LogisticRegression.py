import os
from PIL import Image 
from PIL import ImageFilter
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import random
train_images = os.listdir("MyTrain")
test_images = os.listdir("MyTest")
print ("Training Start...")
clf = LogisticRegression()
X=[]
Y=[]
test_x = []
test_y = []
train_dogs=0
train_cats=0
for each in tqdm(train_images):
    image_data = Image.open(os.path.join("MyTrain",each)).convert('L')
    image_resize = image_data.resize((64,64), Image.ANTIALIAS)
    image_filter=image_resize.filter(ImageFilter.GaussianBlur(255))
    image_final=image_filter.histogram() 
    X.append(image_final)
    if ('dog' in each):
        Y.append(1)
        train_dogs+=1
    elif ('cat' in each):
        Y.append(0)
        train_cats+=1

test_dogs =0 
test_cats =0
for each in tqdm(test_images):
    image_data = Image.open(os.path.join("MyTest",each)).convert('L')
    image_resize = image_data.resize((64,64), Image.ANTIALIAS)
    image_filter=image_resize.filter(ImageFilter.GaussianBlur(255))
    image_final=image_filter.histogram() 
    test_x.append(image_final)
    if ('dog' in each):
        test_y.append(1)
        test_dogs +=1
    elif ('cat' in each):
        test_y.append(0)
        test_cats+=1

#Z = list(zip(X, Y))
#random.shuffle(Z)
#X, Y = zip(*Z)
#slice =int((0.2)*len(X))
#test_x = X[-slice:]
#test_y = Y[-slice:]
#X=X[:-slice]
#Y=Y[:-slice]

clf = clf.fit(X,Y)
print ("Model Training Completed")

correct=0
count=0
for each in tqdm(test_x):
        predicted = clf.predict_proba([each])
        if (float(predicted[0][0])>0.5):
            predicted_label=0
        else:
            predicted_label=1
        if (predicted_label==test_y[count]):
            correct+=1
        count+=1

print ("Data Statistics \n\nTraining Data: ("+str(train_dogs+train_cats)+")\n\tDogs: "+str(train_dogs)+"\n\tCats: "+str(train_cats))
print ("\nTesting Data: ("+str(test_cats+test_dogs)+")\n\tDogs: "+str(test_dogs)+"\n\tCats: "+str(test_cats))
print("\nAccuracy",float(correct)/count)
