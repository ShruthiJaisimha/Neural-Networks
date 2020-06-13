#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
torch.__version__
print(torch.cuda.is_available())
t = torch.tensor([1,2,3])
print(t)
print(t.cuda())


# GPU as the hardware. Cuda as the software architecture on top of the hardware and then the cuDNN on top of Cuda. On top of Cuda is PyTorch.
# 
# ###### Tensors - 
# They are the primary data structures that inputs outputs and transformations are all represented using tensors.
# 
# number, array, 2d-array - these are the terms used in Computer Science
# 
# scalar, vector, matrix - these are the terms used in Mathematics
# 
# These above data structures can be linked to each other by the number of incides required to address them.
# 
# Once we start to need more than 2 indices we can such terms as nd Tensor in mathematics and while it is called nd-array in Computer Science.
# 
# ###### Tensor Attributes - 
# 1. Rank - It represents the number of dimensions present within the tensor. 
# 2. Axis - It reveals the length of the axes and this indicates how many indices are avaliable on the axes.
# 3. Shape - It is determined by the length of the axis. through this we will know how many indices are avaliable on the axes.
#     1. The length of the shape represnets the ranks and axis and each indices of the shape represents a specific axis and the value gives the number of indices in the axis.
#     2. [?,?,?,?] - this represnets the shape of the neural network.
#     The last index represents the actual data stored within the tensor.
#     While the rest are multidimensional arrays.
#     Example: For an image input, the data is represented using pixels which are aligned in 2D array. Thus, the last two indices are the Height and Width of the image. [?,?,H,W]
#     3. [?,C,H,W] The next dimension represents the color. 
#     4. [B,C,H,W] The value in the first dimension tells us how many samples in one batch is used. 

# In[19]:


dd = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
]


# In[21]:


t = torch.tensor(dd)
t.shape
print(t.shape)
print(len(t.shape))
x = t.reshape(1,9)
print(x)
print(x.shape)


# In[2]:


import torch 
import numpy as np


# In[6]:


t = torch.Tensor()
type(t)


print(t.dtype)
print(t.device)
print(t.layout)

device = torch.device('cuda:0')
device


# In[16]:


## Tensors are Uniform Computations
## Tensors between tensor computations are dependent on data type an device.

d1 = torch.tensor([1,2,3])
d2 = torch.tensor([1.,2.,3.])

print(d1.dtype)
print(d2.dtype)

print(d1+d2)

d1 = torch.tensor([1,2,3])
d2 = d1.cuda()

print(d1+d2)


# In[24]:


d1 = np.array([1,2,3])
print(torch.tensor(d1))
print(torch.Tensor(d1))
print(torch.as_tensor(d1))
print(torch.from_numpy(d1))


# In[29]:


print(torch.eye(2))
print(torch.zeros(2,2))
print(torch.ones(1,2))
print(torch.rand(3,4))


# In[37]:


data = np.array([1,2,3])
## using 'T' in tensor indicates that it is constructor
t1 = torch.Tensor(data)

'''
Using 't' in tensor indicates a factory function. 
Factory function accept the data and returns a specific typr of data output.
In this function it is tensor. Factory Function in OOP programming is a 
object and it creates other objects.
The other two instructions are also factory functions.

'''

t2 = torch.tensor(data)
t3 = torch.as_tensor(data)
t4 = torch.from_numpy(data)

print(t1.dtype, t2.dtype, t3.dtype, t4.dtype)

print(torch.tensor(np.array([1,2,3]), dtype = torch.float32))


data[0] = 0
data[1] = 0
'''
The first two t1, t2 tensors copy the data while t3, t4 share the memory
with the numpy array.
'''
print(t1)
print(t2)
print(t3)
print(t4)


# In[7]:


import torch

t = torch.tensor([
    [1,1,2,3],
    [2,4,5,6],
    [5,1,2,0]
])

print(t.dtype)
print(len(t.shape))

torch.tensor(t.shape).prod()
t.numel()


# In[20]:


def flatten(t):
    t = t.reshape(1,-1)
    t = t.squeeze()
    return t


# In[21]:


x = flatten(t)
print(x)


# In[26]:


import torch


t1 = torch.tensor([
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1]
])



t2 = torch.tensor([
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2]
])



t3 = torch.tensor([
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3]
])


# In[3]:


t = torch.stack((t1,t2,t3))
t.shape


# In[4]:


t


# In[9]:


t = t.reshape(3,1,4,4)
t


# In[11]:


t = t.reshape(1,-1)[0]
t


# In[12]:


t.reshape(-1)


# In[13]:


t.view(t.numel())


# In[17]:


t.flatten()


# In[21]:


t.flatten(start_dim=0).shape


# In[23]:


t.flatten(start_dim=0)


# ###### Element wise tensor operation for deep learning
# 
# Element wise operation is an operation between two tensors that operates on corresponding elements within the respective tensors.
# 
# It must have the same shape
# 
# The tensor broadcasting helps in making the tensor shape same as that of the other tensor. 

# In[33]:


t1 = torch.tensor([
    [2,1],
    [2,0]
])

t2 = torch.tensor([
    [5,3],
    [3,8]
])


# In[34]:


t1+t2


# ##### CNN Image Preparation
# 
# 1. Prepare the data
# 2. Build the model
# 3. Train the model
# 4. Analyze the model
# 
# ##### Prepare the data
# ETL Process
# 1. Extract - Get the imgaes from the MNIST dataset
# 2. Transform - Put our data into a tensor form
# 3. Load - Put our data into an object to make it easily accessible

# In[35]:


import torch
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST'
    , train = True
    , download = True
    , transform = transforms.Compose([
        transforms.ToTensor()
    ])
)


train_loader = torch.utils.data.DataLoader(
    train_set, batch_size = 20
)


# In[36]:


import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth = 120)

sample = next(iter(train_set))
image, label = sample

batch = next(iter(train_loader))
images, labels = batch

plt.imshow(image.squeeze(), cmap = 'gray')

grid = torchvision.utils.make_grid(images, nrow = 100)

plt.figure(figsize = (15,15))
plt.imshow(np.transpose(grid,(1,2,0)))


# ###### OOPs
# When writing the program, there are two key components i. Code ii. Data.
# With OOPs we orient our programming around objects.
# 
# 
# Objects - They ar defined in code using classes. A class defines the objects specification which specifies what data and code each object class should have. We call the objects an instance of the class and all instances of a given class have two components.
#     1. Methods - They represent the code. They can tell what the object can do. Ex: A car can move back, front, turn, accelerate.
#     2. Attributes - They represent the attributes. They can tell the characteristics of the object. Ex: A car has this specific length, color. 
#     
# Parameters - They are used within the functions. They can be considered as placeholders.
#     1. Hyperparamters - Their values are chosen manually or arbitrarily. As neural network programmers, we choose hyperparameter values mainly based on trial and error and increasingly by utilizing values that have proven to work in the past. 
#     2. Data dependent hyperparameters - They are the parameters which depend on the data. 
# 
# 
# Arguments - They are the actual values that are passed into a function when it is called.
#     
# ###### Kernal can be interchanged with filter. 
# kernal_size indicates that number of filters that will be used to convolve the given input image.
# ###### out_channels - feature maps. 
# We increase the number of output channels with the increase in the number of convolutional layers and shrink the layers with the increase in the linear layers.
# 
# 
# 
# ##### Build the Model
# 
# Model refers to the neural network
# 
# 

# In[122]:


import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels = 6, kernel_size =5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels = 12, kernel_size =5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        
        return t
    


# In[ ]:





# In[71]:





# In[ ]:




