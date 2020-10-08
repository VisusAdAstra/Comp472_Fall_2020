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
import sklearn.preprocessing         # For scale function
import sklearn.metrics as metrics    # Fo# for label size


# In[21]:


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


# In[20]:


#normalize
from sklearn.preprocessing import StandardScaler
data1.replace('?', np.nan, inplace= True)
data1 = data1.astype({"avg": np.float64})
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# In[81]:


#GaussianNB
from sklearn.naive_bayes import GaussianNB
model1 = GaussianNB()
model2 = GaussianNB()
process("GNB")


# In[34]:


#Baseline Decision Tree
from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier(criterion="entropy")
model2 = DecisionTreeClassifier(criterion="entropy")
process("Base-DT")


# In[28]:


#Best Decision Tree
from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=2, min_impurity_decrease=0.0, class_weight="balanced")
model2 = DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=2, min_impurity_decrease=0.0, class_weight="balanced")
process("Best-DT")


# In[35]:


#Perceptron
from sklearn.linear_model import Perceptron
model1 = Perceptron()
model2 = Perceptron()
process("PER")


# In[82]:


#Base MLP
from sklearn.neural_network import MLPClassifier
model1 = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', solver='sgd')
model2 = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', solver='sgd')
process("Base-MLP")


# In[84]:


#Best MLP
#model = MLPClassifier(hidden_layer_sizes=(150,100,50), activation='relu', learning_rate_init=0.01, max_iter=500,  solver='adam', random_state=1)
from sklearn.neural_network import MLPClassifier
model1 = MLPClassifier(hidden_layer_sizes=(150,100,50), activation='relu', solver='adam', random_state=1)
model2 = MLPClassifier(hidden_layer_sizes=(150,100,50), activation='relu', solver='adam', random_state=1)
process("Best-MLP")


# In[64]:


#validation
def validation(index):
    """
    """
    if index==1:
        X_train = data1.values[:,:-1]
        Y_train = data1.values[:,-1:]
        model1.fit(X_train, Y_train)
        X_val = vali1.values[:,:-1]
        Y_val = vali1.values[:,-1:]
        Y_pred = model1.predict(X_val)
    elif index==2:
        X_train = data2.values[:,:-1]
        Y_train = data2.values[:,-1:]
        model2.fit(X_train, Y_train)
        X_val = vali2.values[:,:-1]
        Y_val = vali2.values[:,-1:]
        Y_pred = model2.predict(X_val)

    print(metrics.accuracy_score(Y_val, Y_pred))


# In[60]:


#test
def test(index):
    """
    """
    if index==1:
        X_test = test1.values[:,:-1]
        Y_test = test1.values[:,-1:]
        Y_pred = model1.predict(X_test)
    elif index==2:
        X_test = test2.values[:,:-1]
        Y_test = test2.values[:,-1:]
        Y_pred = model2.predict(X_test)
    
    print(metrics.accuracy_score(Y_test, Y_pred))


# In[47]:


#save
def save(name, index):
    """
    """
    if index==1:
        Input = samp1.values
        Output = model1.predict(Input)
    elif index==2:
        Input = samp2.values
        Output = model2.predict(Input)
    
    dict = {'Output': Output}  
    df = pd.DataFrame(dict) 
    df.to_csv(name + "-DS" + str(index) + ".csv") 


# In[80]:


#evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn
def evaluation(index):
    """
    """
    if index==1:
        X_val = vali1.values[:,:-1]
        Y_val = vali1.values[:,-1:]
        Y_pred = model1.predict(X_val)
    elif index==2:
        X_val = vali2.values[:,:-1]
        Y_val = vali2.values[:,-1:]
        Y_pred = model2.predict(X_val)

    confusion = confusion_matrix(Y_val, Y_pred)
    print('Confusion Matrix\n')
    plt.figure(figsize = (20,20))
    sn.set(font_scale=1) # for label size
    sn.heatmap(confusion, annot=True, annot_kws={"size": 12}, cmap='Oranges', fmt='d') # font size
    plt.show()

    print('\nClassification Report\n')
    print(classification_report(Y_val, Y_pred))


# In[83]:


#process
def process(name):
    """
    """
    validation(1)
    test(1)
    save(name, 1) 
    validation(2)
    test(2)
    save(name, 2)
    evaluation(1)
    evaluation(2)


# In[56]:




