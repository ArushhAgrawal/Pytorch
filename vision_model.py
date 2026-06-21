import torch
from torch import nn
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy as np
import torchvision 
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

#getting a dataset FashionMnist 
train_data= datasets.FashionMNIST(root= "image_data",
                                  train=True,#we want which type of dataset 
                                  download=True,
                                  transform=transforms.ToTensor(),
                                  target_transform=None)
test_data=datasets.FashionMNIST(root="image_data",
                                train=False,
                                download=True,
                                transform=transforms.ToTensor(),
                                target_transform=None)
header= train_data.classes
header_idx= train_data.class_to_idx
print(f"header index: {header_idx}")
images, labels= train_data[8]
print(f"image shape: {images.shape}")#colour channel, height, width 

#visulisation
# plt.figure(figsize=(10,7))
# plt.imshow(images.squeeze())
# plt.show()

#ploting random image
torch.manual_seed(32)
fig =  plt.figure(figsize=(9,9))
r,c =4,3
label=[]
for i in range(1,r*c+1):
    random_idx=torch.randint(0, len(train_data), size=[1]).item()#0 is starting range and len(train_data) is the ending,  size converts data into tensor with 1 value and item pops out that number as a regulat int
    images ,label= train_data[random_idx]
    fig.add_subplot(r,c,i)#allows to plot multiple images in a single graph sheet
    plt.imshow(images.squeeze())
#plt.show()

#prepare data loader
#it converts our data from pytorch tensor to python itrable
#we will convert our data to batch size (it will reduce the load on ram)
train_dataloader= DataLoader(dataset=train_data,
                             batch_size=32,
                             shuffle=True)
test_dataloader= DataLoader(dataset=test_data,
                            batch_size=32,
                            shuffle=True)
print(f"loaded data: {train_dataloader}, {test_dataloader}")
train_feature, train_label= next(iter(train_dataloader))
test_feature, test_label= next(iter(test_dataloader))

#build baseline model- a bseline model is a simple model we improve using subsiquent model/ experiments

#creating a flatten layer
flatten_model= nn.Flatten()
#get a single sample
x=train_feature
x_flat= flatten_model(x)
print(x_flat.shape)
x_sq=torch.squeeze(x_flat).squeeze(1)
print(x_sq)
