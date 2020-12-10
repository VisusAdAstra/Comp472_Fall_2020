import torch
import numpy as np
import pandas as pd 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import defaultdict
from scipy.special import softmax

import importlib
from . import utils as util
importlib.reload(util)


class EnsembleModel(torch.nn.Module):
    def __init__(self, embeddings_tensor,
                 hidden_size=256,
                 dropout=.5,
                 embedding_size=300, ):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(embeddings_tensor)
        self.lstm = torch.nn.LSTM(embedding_size,
                                  hidden_size,
                                  batch_first=True,
                                  bidirectional=False,
                                  num_layers=3,
                                  dropout=dropout)
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        output = self.embedding(input)
        _, hidden = self.lstm(output)
        hidden = hidden[0]
        output = self.linear(hidden[-1])
        output = self.sigmoid(output)
        return output


class TweetDataset(torch.utils.data.Dataset):
    """
    Inherited class used to fetch datapoints from the dataset and return them as model-ready tuples of tweets and
    annotations
    """
    def __init__(self, ds_loc, embeddings_model):
        self.df = pd.read_csv(ds_loc, sep="\t")
        self.lexicon = Lexicon(embeddings_model)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tweet = self.df.iloc[index, 1]
        tweet = tensorFromSentence(self.lexicon, tweet)
        tweet_id = self.df.iloc[index, 0]
        annotation = self.df.iloc[index, 2]
        annotation = [1] if annotation == 'yes' else [0]
        annotation = torch.tensor(annotation, dtype=torch.long)
        return tweet, annotation, tweet_id


class NB_BOW:
    def __init__(self, unique_classes, filter = False):
        # Constructor is sinply passed with unique number of classes of the training set
        self.classes = unique_classes
        self.smoothing = 0.01
        self.filter = filter

    def addToBow(self, document, dict_index):
        '''
            dict_index = class_index
        '''
        if isinstance(document, np.ndarray):
            document = document[0]

        for token_word in document.split():
            self.bow_dicts[dict_index][token_word] += 1

    def filterWord(self):
        '''
            dict_index = class_index
        '''
        for dict_index in range(self.bow_dicts.shape[0]):
            for token_word in self.bow_dicts[dict_index].copy():
                if(self.bow_dicts[dict_index][token_word] == 1):
                    self.bow_dicts[dict_index].pop(token_word, None)
                
    def train(self, dataset, labels):
        '''
            Parameters:
            1. dataset - shape = (m X d)
            2. labels - shape = (m,)        
        '''
        self.docs = dataset
        self.labels = labels
        self.bow_dicts = np.array([defaultdict(lambda:0)
                                   for index in range(self.classes.shape[0])])

        #validate input format
        if not isinstance(self.docs, np.ndarray):
            self.docs = np.array(self.docs)
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)

        #constructing BoW for each category
        for index, cat in enumerate(self.classes):
            #filter all docs of category == cat
            all_cat_docs = self.docs[self.labels == cat]

            #get docs preprocessed
            cleaned_docs = [util.preProcess(cat_doc)
                            for cat_doc in all_cat_docs]
            cleaned_docs = pd.DataFrame(data=cleaned_docs)

            #now costruct BoW of this particular category
            np.apply_along_axis(self.addToBow, 1, cleaned_docs, index)

        #print(self.bow_dicts[1]['indians'])
        print(len(self.bow_dicts[1]))
        print(len(self.bow_dicts[0]))
        if (self.filter == True):
            with open("test.txt", "w") as file: 
                file.write(str(self.bow_dicts))
            self.filterWord()
        #print(self.bow_dicts[1]['indians'])
        print(len(self.bow_dicts[1])) 
        print(len(self.bow_dicts[0]))       

        '''
            ------------------------------------------------------------------------------------
            Test Time Forumla: {for each word w [ count(w|c)+1 ] / [ count(c) + |V| + 1 ] } * p(c)
            ------------------------------------------------------------------------------------
            
            We are done with constructing of BoW for each category. But we need to precompute a few 
            other calculations at training time too:
            1. prior probability of each class - p(c)
            2. vocabulary |V| 
            3. denominator value of each class - [ count(c) + |V| + 1 ] 
            
        '''

        prob_classes = np.empty(self.classes.shape[0])
        all_words = []
        cat_word_counts = np.empty(self.classes.shape[0])
        for index, cat in enumerate(self.classes):

            #Calculating prior probability p(c) for each class
            prob_classes[index] = np.sum(
                self.labels == cat)/float(self.labels.shape[0])

            #Calculating total counts of all the words of each class
            count = list(self.bow_dicts[index].values())
            cat_word_counts[index] = np.sum(np.array(list(
                self.bow_dicts[index].values()))) + self.smoothing  # |v| is remaining to be added

            #get all words of this category
            all_words += self.bow_dicts[index].keys()

        #combine all words of every category & make them unique to get vocabulary -V- of entire training set

        self.vocab = np.unique(np.array(all_words))
        self.vocab_length = self.vocab.shape[0]

        #computing denominator value
        denoms = np.array([cat_word_counts[index]+self.vocab_length +
                           self.smoothing for index, cat in enumerate(self.classes)])

        '''
            Now that we have everything precomputed as well, its better to organize everything in a tuple 
            rather than to have a separate list for every thing.

            Every element of self.cats_info has a tuple of values
            Each tuple has a dict at index 0, prior probability at index 1, denominator value at index 2
        '''

        self.cats_info = [(self.bow_dicts[index], prob_classes[index], denoms[index])
                          for index, cat in enumerate(self.classes)]
        self.cats_info = np.array(self.cats_info)

    def getDocProb(self, test_doc):
        '''
            Probability of test example in ALL CLASSES
        '''

        # to store probability w.r.t each class
        likelihood_prob = np.zeros(self.classes.shape[0])

        #finding probability w.r.t each class of the given test example
        for index, cat in enumerate(self.classes):
            #for each word w [ count(w|c)+1 ] / [ count(c) + |V| + 1 ]
            #split the test example and get p of each test word
            for test_token in test_doc.split():
                #get total count of this test token from it's respective training dict to get numerator value
                test_token_counts = self.cats_info[index][0].get(test_token, 0) + 1*self.smoothing # + 1 laplace smoothing

                #now get likelihood of this test_token word
                test_token_prob = test_token_counts / float(self.cats_info[index][2])
                likelihood_prob[index] += np.log(test_token_prob)

        # we have likelihood estimate of the given example against every class but we need posterior probility
        post_prob = np.empty(self.classes.shape[0])
        for index, cat in enumerate(self.classes):
            post_prob[index] = likelihood_prob[index] + np.log(self.cats_info[index][1])

        return post_prob

    def predict(self, test_set):
        '''
            Determines probability of each test example against all classes and predicts the label
            against which the class probability is maximum
        '''

        predictions = []  # to store prediction of each test example
        probs = np.zeros((len(test_set), 2))
        for index, doc in enumerate(test_set):
            #preprocess the test example the same way we did for training set exampels
            cleaned_doc = util.preProcess(doc[0])

            #get the posterior probability for both classes
            post_prob = self.getDocProb(cleaned_doc)
            probs[index] = softmax(post_prob/(post_prob[0]+ post_prob[1]))

            #simply pick the max value and map against self.classes!
            predictions.append(self.classes[np.argmax(post_prob)])

        return np.array(predictions), probs

