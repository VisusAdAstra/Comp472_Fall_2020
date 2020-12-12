import numpy as np
from pandas import read_csv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import importlib
from . import models as model
importlib.reload(model)


# ref:https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67
# https://towardsdatascience.com/unfolding-na%C3%AFve-bayes-from-scratch-2e86dcae4b01
# https://github.com/ssghule/Naive-Bayes-and-Bag-of-Words-for-Text-Classification-Problem/blob/master/geolocate.py
def exportData(name, index, data, heu=-1):
    subname = "-h" + str(heu) if heu > -1 else ""
    solution = f"output/{index}_{name}{subname}" + '_solution.txt'
    search = f"output/{index}_{name}{subname}" + '_search.txt'
    file1 = open(solution, 'w')
    file2 = open(search, 'w')
    if isinstance(data[0], node.Node):
        for ele in data[0].solution:
            file1.write(ele)
        file1.write(f"{data[0].gn} {data[2]}")
        for ele in data[1]:
            file2.write(ele)
    else:  
        file1.write(data[0])
        file2.write(data[1])
    file1.close()
    file2.close()
    

def preProcess(str_arg):
    '''
        Return the preprocessed string in tokenized form
    '''
    cleaned_str=re.sub('[^a-z0-9-\s]+',' ',str_arg,flags=re.IGNORECASE) #every char except alphabets is replaced
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) #multiple spaces are replaced by single space
    cleaned_str=cleaned_str.lower() #converting the cleaned string to lower case
    return cleaned_str 


def displayResult(model, X_test, y_test):
    '''
        Display temporary result
    '''
    y_pred, post_prob = model.predict(X_test)
    test_acc=np.sum(y_pred == y_test.reshape(-1))/float(y_test.shape[0]) 

    print ('----------------- Result ---------------------')
    print ("Test Set Size:     ",y_test.shape[0])
    print ("Test Set Accuracy: ",test_acc*100,"%")
    print(y_pred)
    print(y_test.reshape(-1))
    #print(post_prob)


def processData(models, X_test, y_test, dataset):
    '''
        Main process control
    '''
    for model in models:
        displayResult(model, X_test, y_test)
        prec, rec, f1 = statModel(model, X_test, y_test, dataset, True)
        evalModel(model, X_test, y_test)
        print("TEST: Precision: {0:.4}\tRecall: {1:.4}\tF1: {2:.4}".format(prec, rec, f1))    


def statModel(model, X_test, y_test, dataset=None, out=False):
    '''
    Returns evaluation metrics
    :param target: tensor containing target values
    :param prediction: tensor containing prediction values
    :param dataset: (optional, required to produce trace file) test dataset
    :param out: (optional, required to produce trace file) whether to produce trace or not
    :return: tuple containing precision, recall, and f1 values
    '''
    dataset = np.array(dataset)
    y_pred, post_prob = model.predict(X_test)
    if(model.filter == False):
        filename = "result/trace_NB-BOW-OV.txt "
    else:
        filename = "result/trace_NB-BOW-FV.txt"
    if out is not False:
        if dataset is None:
            raise ValueError("Dataset is needed to retrieve tweet ids!")
        with open(filename, "w") as file:
            for i in range(len(dataset)):
                tweet_id = dataset[i][0]
                prediction_text = "yes" if y_pred[i] == 1 else "no"
                prediction_proba = post_prob[i][1].item() if prediction_text == "yes" else post_prob[i][0].item()
                target_text = "yes" if y_test[i] == 1 else "no"
                outcome = "correct" if prediction_text == target_text else "wrong"
                line = """{}  {}  {:.4}  {}  {}\n""".format(tweet_id, prediction_text, prediction_proba, target_text, outcome)
                file.write(line)
        print(f"Trace file produced: '{filename}'")
    try:
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    except ValueError:
        f1 = pre = rec = 0
    return pre, rec, f1


def evalModel(model, X_test, y_test):
    '''
    Produces output file containing model evaluation
    :param prediction: tensor containing prediction values
    :param target: tensor containing target values
    :return: None
    '''
    y_pred, post_prob = model.predict(X_test)
    t_yes = y_test
    t_no = np.array([(lambda x: 1 if x == 0 else 0)(n) for n in t_yes])
    p_yes = y_pred
    p_no = np.array([(lambda x: 1 if x == 0 else 0)(n) for n in p_yes])
    if(model.filter == False):
        filename = "result/eval_NB-BOW-OV.txt "
    else:
        filename = "result/eval_NB-BOW-FV.txt"
    try:
        pre_yes = precision_score(t_yes, p_yes)
        rec_yes = recall_score(t_yes, p_yes)
        f1_yes = f1_score(t_yes, p_yes)
        acc = accuracy_score(t_yes, p_yes)

        pre_no = precision_score(t_no, p_no)
        rec_no = recall_score(t_no, p_no)
        f1_no = f1_score(t_no, p_no)
    except ValueError:
        f1_yes = pre_yes = rec_yes = 0
        f1_no = pre_no = rec_no = 0
    with open(filename, "w") as file:
        file.write("{:.4}\n".format(acc))
        file.write("{:.4}  {:.4}\n".format(pre_yes, pre_no))
        file.write("{:.4}  {:.4}\n".format(rec_yes, rec_no))
        file.write("{:.4}  {:.4}\n".format(f1_yes, f1_no))
    print(f"Evaluation file produced: '{filename}'")

