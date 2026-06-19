import torch
from torch import nn
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy
import sklearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

num_class=4
num_features=2


#create multiclass data
x_blob,  y_blob = make_blobs(n_samples=1000,
                             n_features=num_features,
                             centers= num_class,
                             cluster_std=1.5, #std is standerd devistion gives the data a little shakeup a mix
                             random_state=42)

#turn into tensors
x_blob, y_blob= torch.Tensor(x_blob).to(torch.float32), torch.from_numpy(y_blob).to(torch.float32)

#spliting data in tran and test
x_train, x_test, y_train, y_test = train_test_split(x_blob,
                                                    y_blob,
                                                    test_size=0.8,
                                                    random_state=32)

#device agnostic code
device="mps" if torch.mps.is_available() else "cpu"
print(device)
#visualisation
plt.figure(figsize=(10,7))
plt.scatter(x_blob[:,0], x_blob[:,1], c=y_blob)
#plt.show()

#making model
model_0= nn.Sequential(nn.Linear(in_features=2, out_features=16),
                       nn.ReLU(),
                       nn.Linear(in_features=16, out_features=8),
                       nn.ReLU(),
                       nn.Linear(in_features=8, out_features=4),
                       nn.ReLU(),
                       nn.Linear(in_features=4, out_features=2)).to(device)
print(model_0)