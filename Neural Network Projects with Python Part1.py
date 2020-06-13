#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd C:\Shruthi\Udemy_Course_Mine\Neural-Network-Projects-with-Python-master


# In[2]:


conda env create -f environment.yml


# In[3]:


conda activate neural-network-projects-python


# In[5]:


cd C:\Shruthi\Udemy_Course_Mine\Neural-Network-Projects-with-Python-master\Chapter01


# In[6]:


import keras_chapter1


# In[7]:


import numpy as np

class NeuralNetwork:
    def __init__(self, x, y):
        self.input    = x
        self.weights1 = np.random.rand(self.input.shape[1],4) 
        self.weights2 = np.random.rand(4,1) 
        self.y        = y
        self.output = np.zeros(self.y.shape)
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, seld.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))


# In[8]:


import numpy as np

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
   return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input    = x
        self.weights1 = np.random.rand(self.input.shape[1],4) 
        self.weights2 = np.random.rand(4,1) 
        self.y        = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find the derivation of the 
        # loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) *                                                                          
                     sigmoid_derivative(self.output)))       
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) 
                    * sigmoid_derivative(self.output), self.weights2.T) *                                                
                      sigmoid_derivative(self.layer1))) 

        self.weights1 += d_weights1
        self.weights2 += d_weights2
        #print(self.weights1, d_weights1)
if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[0],[1],[1]])
    nn = NeuralNetwork(X,y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(nn.output)


# In[10]:


import pandas as pd
URL =     'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(URL, names = ['sepal_length', 'sepal_width', 
                           
                               
                               'petal_length', 'petal_width', 'class'])
df


# In[11]:


df.describe()


# In[12]:


df.loc[df['sepal_length'] > 5.0]


# In[13]:


marker_shapes = ['.', '*', '^']
for i, species in enumerate(df['class'].unique()):
    ax = plt.axes()
    x1 = df[df['class'] == species]
    x1.plot.scatter(x='petal_length', y='petal_width', marker = marker_shapes[i], s=50, ax=ax)

df['petal_length'].plot.hist()
x1.plot.scatter(x = 'sepal_width', y='sepal_length')
df.plot.box()


# In[14]:


x2 = pd.DataFrame(['monday', 'tuesday', 'wednesday', 'thursday'])
pd.get_dummies(x2)


# In[193]:


random_index = np.random.choice(df.index, replace= False, size=10)
df.loc[random_index,'sepal_length'] = None
db = df.dropna()

df.sepal_length=df.sepal_length.fillna(df.sepal_length.mean())


# In[194]:


db.info()


# In[195]:


df.info()


# In[206]:


from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
model = Sequential()

model.add(Dense(units=4, activation='sigmoid', input_dim=3))
model.add(Dense(units=1, activation='sigmoid'))
print(model.summary())

model.compile(loss='mean_squared_error', optimizer=optimizers.SGD(learning_rate=0.01))
          


# In[207]:


import numpy as np
# Fixing a random seed ensures reproducible results
np.random.seed(9)

X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],[1],[1],[0]])


# In[208]:


model.fit(X, y, epochs=1500)


# In[209]:


print(model.predict(X))


# In[ ]:




