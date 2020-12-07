import torch
import numpy as np
from pandas import read_csv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from . import models as model


# ref:https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67
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
    """"
        Return the preprocessed string in tokenized form
    """
    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) #every char except alphabets is replaced
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) #multiple spaces are replaced by single space
    cleaned_str=cleaned_str.lower() #converting the cleaned string to lower case
    
    return cleaned_str 


def test(prediction, target, dataset=None, out=False):
    """
    Returns evaluation metrics
    :param target: tensor containing target values
    :param prediction: tensor containing prediction values
    :param dataset: (optional, required to produce trace file) test dataset
    :param out: (optional, required to produce trace file) whether to produce trace or not
    :return: tuple containing precision, recall, and f1 values
    """
    t = np.array([x.item() for x in target])
    p = np.array([pred(x) for x in prediction])
    if out is not False:
        if dataset is None:
            raise ValueError("Dataset is needed to retrieve tweet ids!")
        with open("eval_NB-BOW-" + id + ".txt", "w") as file:
            for i in range(len(dataset)):
                tweet_id = dataset[i][2]
                prediction_text = "yes" if pred(prediction[i]) == 1 else "no"
                prediction_proba = prediction[i][1].item() if prediction_text == "yes" else prediction[i][0].item()
                target_text = "yes" if target[i] == 1 else "no"
                outcome = "correct" if prediction_text == target_text else "wrong"
                line = """{}  {}  {:.4}  {}  {}\n""".format(tweet_id, prediction_text, prediction_proba, target_text, outcome)
                file.write(line)
        print(f"Trace file produced: 'eval_NB-BOW-{id}.txt'")
    try:
        pre = precision_score(t, p)
        rec = recall_score(t, p)
        f1 = f1_score(t, p)
    except ValueError:
        f1 = pre = rec = 0
    return pre, rec, f1


def evaluateModel(model, prediction, target):
    """
    Produces output file containing model evaluation
    :param model: lstm torch model
    :param prediction: tensor containing prediction values
    :param target: tensor containing target values
    :return: None
    """
    t_yes = np.array([x.item() for x in target])
    t_no = np.array([(lambda x: 1 if x == 0 else 0)(n) for n in t_yes])
    p_yes = np.array([pred(x) for x in prediction])
    p_no = np.array([(lambda x: 1 if x == 0 else 0)(n) for n in p_yes])
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
    with open("eval_NB-BOW-" + model.id + ".txt", "w") as file:
        file.write("{:.4}\n".format(acc))
        file.write("{:.4}  {:.4}\n".format(pre_yes, pre_no))
        file.write("{:.4}  {:.4}\n".format(rec_yes, rec_no))
        file.write("{:.4}  {:.4}\n".format(f1_yes, f1_no))
    print(f"Evaluation file produced: 'eval_NB-BOW-{model.id}.txt'")

