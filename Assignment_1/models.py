#!/usr/bin/env python
# coding: utf-8

# Readme: execute each block by adding #%%
# Python 3.8, Sklearn 0.23.2

#%%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
import sklearn.preprocessing    # For scale function
import sklearn.metrics          # For accuracy_score


# In[115]:


#load
data1 = pd.read_csv('Assig1-Dataset/train_1.csv')
vali1 = pd.read_csv('Assig1-Dataset/val_1.csv')
test1 = pd.read_csv('Assig1-Dataset/test_with_label_1.csv')
samp1 = pd.read_csv('Assig1-Dataset/test_no_label_1.csv')

data2 = pd.read_csv('Assig1-Dataset/train_2.csv')
vali2 = pd.read_csv('Assig1-Dataset/val_2.csv')
test2 = pd.read_csv('Assig1-Dataset/test_with_label_2.csv')
samp2 = pd.read_csv('Assig1-Dataset/test_no_label_2.csv')

info1 = pd.read_csv('Assig1-Dataset/info_1.csv')
info2 = pd.read_csv('Assig1-Dataset/info_2.csv')


# In[ ]:


#normalize
from sklearn.preprocessing import StandardScaler
data1.replace('?', np.nan, inplace= True)
data1 = data1.astype({"avg": np.float64})
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# In[111]:


#GaussianNB
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
validation(1)
validation(2)


# In[112]:


#Baseline Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion="entropy")
validation(1)
validation(2)


# In[113]:


#Best Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=2, min_impurity_decrease=0.0, class_weight="balanced")
validation(1)
validation(2)


# In[136]:


#Perceptron
from sklearn.linear_model import Perceptron
model = Perceptron()
validation(1)


# In[101]:


#Base MLP
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', solver='sgd')
validation(1)


# In[135]:


#Best MLP
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(150,100,50), activation='relu', learning_rate_init=0.01, max_iter=500,  solver='adam', random_state=1)
validation(1)


# In[109]:


#validation
def validation(index):
    """
    """
    if index==1:
        X_train = data1.values[:,:-1]
        Y_train = data1.values[:,-1:]
        model.fit(X_train, Y_train)
        X_val = vali1.values[:,:-1]
        Y_val = vali1.values[:,-1:]
    elif index==2:
        X_train = data2.values[:,:-1]
        Y_train = data2.values[:,-1:]
        model.fit(X_train, Y_train)
        X_val = vali2.values[:,:-1]
        Y_val = vali2.values[:,-1:]
    predict_test = model.predict(X_val)
    print(metrics.accuracy_score(Y_val, predict_test))


# In[110]:


#test
def test(index):
    """
    """
    if index==1:
        X_val = test1.values[:,:-1]
        Y_val = test1.values[:,-1:]
    elif index==2:
        X_val = test2.values[:,:-1]
        Y_val = test2.values[:,-1:]
    predict_test = model.predict(X_val)
    print(metrics.accuracy_score(Y_val, predict_test))


# In[137]:


#test1
X_val = test1.values[:,:-1]
Y_val = test1.values[:,-1:]
predict_test = model.predict(X_val)
print(metrics.accuracy_score(Y_val, predict_test))


# In[ ]:


#test2
X_val = test2.values[:,:-1]
Y_val = test2.values[:,-1:]
predict_test = model.predict(X_val)
print(metrics.accuracy_score(Y_val, predict_test))


# In[138]:


#save1
Input = samp1.values
Output = model.predict(Input)
dict = {'Output': Output}  
df = pd.DataFrame(dict) 
df.to_csv('submission_1.csv') 


# In[134]:


#save2
Input = samp2.values
Output = model.predict(Input)
dict = {'Output': Output}  
df = pd.DataFrame(dict) 
df.to_csv('submission_2.csv') 

