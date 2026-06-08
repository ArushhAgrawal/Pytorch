#pytorch workflow
#what work flow has
#1 preparing of data
#2 build model
#3 fitting the model to data or training
#4 making prediction and making changes in weights
#5 saving and repeating
import torch
from torch import nn #nn contains all the pytorch building block from nural network
import matplotlib
matplotlib.use('MACOSX')#this is important if u are on vs code use TkAgg if on windows 
import matplotlib.pyplot as plt

#data preparing and loading
#data can be a lot of things like- Excel sheet, images, videos, audio...etc

#in machine learning - u take data convert to number(vectors) then make computer learn patterns in it.
#to showcase this we create ome leaner regression formula (y=mx+c) to draw a straight line from the known parameters

#create known parameters
weight= 0.7
bias= 0.3

#create data
start=0
end=1
step=0.02
x= torch.arange(start,end,step).unsqueeze(1)#it adds extra dimesnions will be usefull later 
y= weight * x+bias

#print(x[:10])
#print(y[:10])
#print(len(x))
#print(len(y))

#splitting data in training and testing data
#create train,test split
train_split = int(0.8*len(x))
x_train, y_train= x[:train_split], y[:train_split]
x_test, y_test= x[train_split:], y[train_split:]
print(f"traing split:  {train_split}")
print(f"traing x:  {len(x_train)}")
print(f"traing y:  {len(y_train)}")
print(f"testing x:  {len(y_test)}")
print(f"testing y:  {len(y_test)}")

#visualisation of tensor/numbers
def plot_prediction(train_data= x_train,
                    train_label=y_train,
                    test_labels= y_test,
                    test_data= x_test,
                    prediction=None):
    plt.figure(figsize=(10,7))

    #plot training data in blue
    plt.scatter(train_data, train_label, c="b", s=4, label= "training data")

    #plot testing data in green ur choice
    plt.scatter(test_data, test_labels, c='g', label= "test data")

    #are there prediction
    if prediction is not None:
        plt.scatter(test_data, prediction, c= "red", label= "prediction data")
    
    #show the legends (can put a mirror too)
    plt.legend(prop={"size":14})
    plt.show()

plot_prediction()