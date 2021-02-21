import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
nltk.download('stopwords')
nltk.download('punkt')

class corpus_prepare:

    ''' Remove missing cases and split the dataset into a balaned trainset
        and a testset that keeps original ratio from the past.
    '''
    
    def __init__(self, df):
        self.df = df
        self.violation_cases = list(df[df.Judgement == 'violation']['Judgement_text'])
        self.no_violation_cases = list(df[df.Judgement == 'no_violation']['Judgement_text'])
        self.ratio = (len(self.violation_cases)/len(df), len(self.no_violation_cases)/len(df))

    def remove_missing_case(self):
        # drop cases with no judgement doc available
        no_text = self.df[self.df.Judgement_text == 'unavailable'].index
        self.df.drop(no_text, inplace = True)
    
    def train_test_split(self, split_ratio = (0.9, 0.1)):
        # ratio of train_test split
        self.split_ratio = split_ratio
        # keeping the ratio of judgements in the past for test set 
        no_vio_test_num = round(len(self.no_violation_cases) * split_ratio[1])
        total_test_num = no_vio_test_num/self.ratio[1]
        vio_test_num = round(total_test_num * self.ratio[0])
        # balanced training set 50:50
        total_train_num = round(len(self.no_violation_cases) - no_vio_test_num) * 2
        # spliting
        test_no_vio = self.no_violation_cases[(len(self.no_violation_cases) - no_vio_test_num):]
        train_no_vio = self.no_violation_cases[:(len(self.no_violation_cases) - no_vio_test_num)]
        test_vio = self.violation_cases[(len(self.violation_cases) - vio_test_num):]
        train_vio = self.violation_cases[:round(total_train_num/2)] # find the better way to split
        
        # whole_train, test set
        train_x = train_vio + train_no_vio
        test_x = test_vio + test_no_vio
        # combine positive and negative labels
        train_y = np.append(np.ones((len(train_vio), 1)), np.zeros((len(train_no_vio), 1)), axis=0)
        test_y = np.append(np.ones((len(test_vio), 1)), np.zeros((len(test_no_vio), 1)), axis=0)
        
        return train_x, test_x, train_y, test_y
    
class preprocessing_text:

    ''' Text preprocessing steps 

    '''
    
    def __init__(self):
        pass
    
    def remove_stopwords(self, tokenised_corpus, stop_words = 'nltk'):
        tokenised_corpus = [token.lower() for token in tokenised_corpus]
        if stop_words == 'nltk':
            stopwords_english = stopwords.words('english')
        elif stop_words == 'spacy':
            stopwords_english = STOP_WORDS
        else :
            stopwords_english = []
            print('no appropriate library to get stopwords is given')
        
        clean_corpus = [word for word in tokenised_corpus if word not in stopwords_english]
                
        return clean_corpus
    
    def remove_punctuations(self, tokenised_corpus):
        tokenised_corpus = [token.lower() for token in tokenised_corpus]
        punc = string.punctuation + "“”’§§...–``''."
        clean_corpus = [word for word in tokenised_corpus if word not in punc]
        
         #double cleaning for some cases like convention”
        #for word in clean_corpus:
            #clean_corpus.append(re.sub(r"[^a-zA-Z0-9]+", "", word))
                     
        return clean_corpus 
    
    def remove_numbers(self, tokenised_corpus):
        tokenised_corpus = [token.lower() for token in tokenised_corpus]
        clean_corpus = [word for word in tokenised_corpus if not any(c.isdigit() for c in word)]
        
        return clean_corpus