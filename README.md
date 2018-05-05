# DogsVsCats
Dogs vs Cats Image Classification using Convolution Neural Networks
---------------------------------------
+ EE 769 ML Project
+ 173050010 - Sai Prasad
+ 173050050 - Adarsh Rathore


*How to Run:*

+ Download Generate.py, LinearRegression.py and CNN.py
+ Donwload train.zip and test.zip from https://www.kaggle.com/c/dogs-vs-cats
+ Extract both zips
+ Create Folders MyTrain, MyTest, OrgTest (in Current Directory)
+ Run 'python3 Generate.py'
    + Code will convert original test and train images into 64x64 size
    + Code will Split train data into two folders MyTrain and MyTest (with 80:20 ratio)
    + Code will move resized test images into OrgTest folder
    + As we don't having exact labels on test images to predict accuracy, we don't use them.

+ Execute 'python3 LinearRegression.py' 
+ Execute 'python3 CNN.py' 



For any queries:
saiprasad@cse.iitb.ac.in

