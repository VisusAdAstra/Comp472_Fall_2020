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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn

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


#normalize
#from sklearn.preprocessing import StandardScaler
#data1.replace('?', np.nan, inplace= True)
#data1 = data1.astype({"avg": np.float64})
#sc = StandardScaler()
#X_train = sc.fit_transform(x_train)
#X_test = sc.transform(x_test)


################################### Methods ##########################################

#train
def train(index):
    if index==1:
        X_train = data1.values[:,:-1]
        Y_train = data1.values[:,-1:]
        model1.fit(X_train, Y_train)
    elif index==2:
        X_train = data2.values[:,:-1]
        Y_train = data2.values[:,-1:]
        model2.fit(X_train, Y_train)

#validation
def validation(index):
    if index==1:
        X_val = vali1.values[:,:-1]
        Y_val = vali1.values[:,-1:]
        Y_pred = model1.predict(X_val)
    elif index==2:
        X_val = vali2.values[:,:-1]
        Y_val = vali2.values[:,-1:]
        Y_pred = model2.predict(X_val)

    print(metrics.accuracy_score(Y_val, Y_pred))


#test
def test(index):
    if index==1:
        X_test = test1.values[:,:-1]
        Y_test = test1.values[:,-1:]
        Y_pred = model1.predict(X_test)
    elif index==2:
        X_test = test2.values[:,:-1]
        Y_test = test2.values[:,-1:]
        Y_pred = model2.predict(X_test)
    
    print(metrics.accuracy_score(Y_test, Y_pred))


#save
def save(name, index):
    if index==1:
        Input = samp1.values
        Output = model1.predict(Input)
    elif index==2:
        Input = samp2.values
        Output = model2.predict(Input)
    
    dict = {'Output': Output}  
    df = pd.DataFrame(dict) 
    df.to_csv("output file/" + name + "-DS" + str(index) + ".csv") 


#evaluation
def evaluation(name ,index):
    if index==1:
        X_val = vali1.values[:,:-1]
        Y_val = vali1.values[:,-1:]
        Y_pred = model1.predict(X_val)
    elif index==2:
        X_val = vali2.values[:,:-1]
        Y_val = vali2.values[:,-1:]
        Y_pred = model2.predict(X_val)

    confusion = confusion_matrix(Y_val, Y_pred)
    confusionResultStr = "Confusing Matrix - {} {}\n".format(name, index);
    cmtx = pd.DataFrame(confusion)
    cmtx.to_csv("output file/" + "Confusing Matrix - {} DS{}.csv".format(name, index))
    print(confusionResultStr)

    #plt.figure(figsize = (20,20))
    #sn.set(font_scale=1) # for label size
    #sn.heatmap(confusion, annot=True, annot_kws={"size": 12}, cmap='Oranges', fmt='d') # font size
    #plt.show()

    print('\nClassification Report\n')
    report = classification_report(Y_val, Y_pred, output_dict=True)

    #Classification Report to CSV
    df = pd.DataFrame(report).transpose()
    df.to_csv("output file/" + name + "-DS" + str(index) + ".csv", mode='a', header=True)

#process
def process(name):
    train(1)
    validation(1)
    test(1)
    save(name, 1) 
    train(2)
    validation(2)
    test(2)
    save(name, 2)
    evaluation(name, 1)
    evaluation(name, 2)
    
################################### Main ##########################################

#GaussianNB - a
from sklearn.naive_bayes import GaussianNB
model1 = GaussianNB()
model2 = GaussianNB()
process("GNB")


#Baseline Decision Tree - b
from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier(criterion="entropy")
model2 = DecisionTreeClassifier(criterion="entropy")
process("Base-DT")


#Best Decision Tree - c
from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier(criterion="entropy", max_depth=None, min_samples_split=5, min_impurity_decrease=0.0003, class_weight="balanced")
model2 = DecisionTreeClassifier(criterion="entropy", max_depth=None, min_samples_split=10, min_impurity_decrease=0.0, class_weight=None)
process("Best-DT")

#Perceptron - d
from sklearn.linear_model import Perceptron
model1 = Perceptron()
model2 = Perceptron()
process("PER")


#Base MLP - e
from sklearn.neural_network import MLPClassifier
model1 = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', solver='sgd')
model2 = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', solver='sgd')
process("Base-MLP")


#Best MLP - f
from sklearn.neural_network import MLPClassifier
model1 = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', solver='adam', random_state=1)
model2 = MLPClassifier(hidden_layer_sizes=(50,50), activation='tanh', solver='adam', random_state=1)
process("Best-MLP")




