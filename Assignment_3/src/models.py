import torch
import numpy as np
import pandas as pd 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import defaultdict

import utils as util


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


class NaiveBayes:
    
    def __init__(self,unique_classes):
        # Constructor is sinply passed with unique number of classes of the training set
        self.classes=unique_classes 
        

    def addToBow(self,document,dict_index):
        '''
            dict_index = class_index
        '''  
        if isinstance(document,np.ndarray): document=document[0]
     
        for token_word in document.split():          
            self.bow_dicts[dict_index][token_word]+=1
    
            
    def train(self,dataset,labels):
        '''
            Parameters:
            1. dataset - shape = (m X d)
            2. labels - shape = (m,)        
        '''
        self.examples=dataset
        self.labels=labels
        self.bow_dicts=np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])
        
        #only convert to numpy arrays if initially not passed as numpy arrays - else its a useless recomputation
        if not isinstance(self.examples,np.ndarray): 
            self.examples=np.array(self.examples)
        if not isinstance(self.labels,np.ndarray): 
            self.labels=np.array(self.labels)
            
        #constructing BoW for each category
        for index,cat in enumerate(self.classes):
            #filter all examples of category == cat
            all_cat_examples=self.examples[self.labels==cat] 
            
            #get examples preprocessed
            cleaned_examples=[util.preprocess(cat_example) for cat_example in all_cat_examples]
            cleaned_examples=pd.DataFrame(data=cleaned_examples)
            
            #now costruct BoW of this particular category
            np.apply_along_axis(self.addToBow,1,cleaned_examples,index)
            
        
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
    
        smoothing = 0.01
        prob_classes=np.empty(self.classes.shape[0])
        all_words=[]
        cat_word_counts=np.empty(self.classes.shape[0])
        for index,cat in enumerate(self.classes):
           
            #Calculating prior probability p(c) for each class
            prob_classes[index]=np.sum(self.labels==cat)/float(self.labels.shape[0]) 
            
            #Calculating total counts of all the words of each class 
            count=list(self.bow_dicts[index].values())
            cat_word_counts[index]=np.sum(np.array(list(self.bow_dicts[index].values()))) + smoothing # |v| is remaining to be added
            
            #get all words of this category                                
            all_words+=self.bow_dicts[index].keys()
                                                     
        
        #combine all words of every category & make them unique to get vocabulary -V- of entire training set
        
        self.vocab=np.unique(np.array(all_words))
        self.vocab_length=self.vocab.shape[0]
                                  
        #computing denominator value                                      
        denoms=np.array([cat_word_counts[index]+self.vocab_length + smoothing for index,cat in enumerate(self.classes)])                                                                          
      
        '''
            Now that we have everything precomputed as well, its better to organize everything in a tuple 
            rather than to have a separate list for every thing.

            Every element of self.cats_info has a tuple of values
            Each tuple has a dict at index 0, prior probability at index 1, denominator value at index 2
        '''
        
        self.cats_info=[(self.bow_dicts[index],prob_classes[index],denoms[index]) for index,cat in enumerate(self.classes)]                               
        self.cats_info=np.array(self.cats_info)                                 
                                              
                                              
    def getDocProb(self,test_doc):                                
        
        '''
            Probability of test example in ALL CLASSES
        '''                                      
                                              
        likelihood_prob=np.zeros(self.classes.shape[0]) #to store probability w.r.t each class
        
        #finding probability w.r.t each class of the given test example
        for index,cat in enumerate(self.classes): 
            #for each word w [ count(w|c)+1 ] / [ count(c) + |V| + 1 ]      
            #split the test example and get p of each test word               
            for test_token in test_doc.split(): 
                #get total count of this test token from it's respective training dict to get numerator value                           
                test_token_counts=self.cats_info[index][0].get(test_token,0)+1
                
                #now get likelihood of this test_token word                              
                test_token_prob=test_token_counts/float(self.cats_info[index][2])                              
                likelihood_prob[index]+=np.log(test_token_prob)
                                              
        # we have likelihood estimate of the given example against every class but we need posterior probility
        post_prob=np.empty(self.classes.shape[0])
        for index,cat in enumerate(self.classes):
            post_prob[index]=likelihood_prob[index]+np.log(self.cats_info[index][1])                                  
      
        return post_prob
    
   
    def test(self,test_set):
        '''
            Determines probability of each test example against all classes and predicts the label
            against which the class probability is maximum
        '''       
       
        predictions=[] #to store prediction of each test example
        for doc in test_set: 
                                              
            #preprocess the test example the same way we did for training set exampels                                  
            cleaned_doc=util.preprocess(doc) 
             
            #simply get the posterior probability of every example                                  
            post_prob=self.getExampleProb(cleaned_doc) #get prob of this example for both classes
            
            #simply pick the max value and map against self.classes!
            predictions.append(self.classes[np.argmax(post_prob)])
                
        return np.array(predictions)

