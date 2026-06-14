#nural network classification in pytoch
# classification is a problem wheather something is one or another
import sklearn
from sklearn.datasets import make_circles as mc
import pandas as pd
import torch
from torch import nn
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#making of data
#making 1000 samples (can me more as required)
n_samples= 1000
x, y = mc(n_samples, 
          noise = 0.02, #randomness
          random_state= 32)
print(f"length of x and y {len(x), len(y)}")
print(f"first 5 samples of x: {x[0:5]}")
print(f"first 5 samples of y: {y[0:5]}")

#make dataframe of circle data
circles = pd.DataFrame({"x1":x[999,0],#just for concept (: means all and 999 means all too since there are 1000 only)
                        "x2":x[:, 1],
                        "labels":y[999,]})#used y.shape to find the shape if we dont use bar bracker it would still show us y
print(circles[:10])

#lets visualize 
plt.scatter(x=x[:,0],
            y=x[:,1],
            c="b",
            label="data")
#plt.show() 
#the dataset we are working on is for practice its called toy dataset too weak 

#sample shape
y_sample=y[0]
x_sample= x[0,:]
print(f"values of 1 the sample of {x_sample}, {y_sample}") 
print(f"size of  the sample of {x_sample.shape}, {y_sample.shape}")
print(x.dtype)
#turning data into tensors
x= torch.from_numpy(x)
#print(x.dtype)
y= torch.from_numpy(y)
#print(y.dtype) 

#spliting data into training and test set
x_train,  x_test,  y_train, y_test = train_test_split(x,#the data
#this order of x train x test... should be same 
                                                      y,#data label can be 0,1 for binary and many othes for multiclas
                                                      train_size= 0.8,#80% goes to train and rest to train automatically
                                                      random_state=42 #very similar to random seed but its for sklearn not torch
                                                      )
#print(len(x_test), len(x_train),len(y_test),len(y_train))

#building of nural network / model building
#lets build a model to classify our blue dots
# to do so we need to
#set up device agnostic code gpu/cpu
#consturct a model (by subclassing nn.module)
#define a loss func and optimizer
#create a training / test loop

device= "mps" if torch.mps.is_available() else "cpu"
print(device)