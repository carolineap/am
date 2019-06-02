import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
import scipy.stats as stats
from scipy.stats import chi2_contingency
from nltk import FreqDist
from nltk.util import ngrams # function for making ngrams
from collections import Counter
from nltk.stem import WordNetLemmatizer 
  
NEGATION = ["not", "no", "nothing", "never", "none"]

def remove_not_alpha(words):
    
    only_alpha = [word.replace("'", "o") for word in words if word.isalpha() or (word == "n't")]
    
    return only_alpha
    
def remove_stop_words(words):
    
    stop_words = stopwords.words('english')
        
    without_sw = [word for word in words if word not in stop_words or word in NEGATION]
    
    return without_sw

def stemming(words):
    
    stemmer = PorterStemmer() 
   
    stemming_words = [ stemmer.stem(word) for word in words]
    
    return stemming_words

def lemmatization(words):
    
    lemmatizer = WordNetLemmatizer() 
    
    lemmatizer_words = [ lemmatizer.lemmatize(word) for word in words]
    
    return lemmatizer_words

def pos_tag(words):
        
    pt_words = []
       
    i = 0
    
    adj = False
    
    for token, pos in nltk.pos_tag(words):
        
        if pos[0] == 'V' or pos[0] == 'J' or pos[0] == 'R': #get only verbs, adverbs and adjectives
            pt_words.append(token)
        
    return pt_words
    
#modifiquei apenas para testar argumentos
def clear_text(words, stem = True, pos_t = True, lemm = False, rm_stop = True, rm_not_alpha = True):

    if (rm_not_alpha):
        words = remove_not_alpha(words)

    if (rm_stop):
        words = remove_stop_words (words)
        
    if (pos_t):
        words = pos_tag (words)
        
    if (lemm):
        words = lemmatization (words)
        
    if (stem):
        words = stemming (words)
        
    return words
         

def handle_negation(words):
           
    with_negation = []
    
    for i in range(len(words)):
        
        if words[i] in NEGATION and i+1 < len(words):
            with_negation.append((words[i], words[i+1]))
            i += 1
        else:
            with_negation.append(words[i])
            
    return with_negation

#modifiquei apenas para testar argumentos
def bow(category, clear = True, stem = True, hand_neg = True, pos_t = True, lemm = False, rm_stop = True, rm_not_alpha = True):
           
    category += '/'

    positive = open('sorted_data_acl/' + category  + 'positive.review', 'r')
    negative = open('sorted_data_acl/' + category  + 'negative.review', 'r')

    positive_reviews = (BeautifulSoup(positive, 'lxml'))
    negative_reviews = (BeautifulSoup(negative, 'lxml'))

    positive_reviews = positive_reviews.find_all(['review'])
    negative_reviews = negative_reviews.find_all(['review'])

    n_pos_reviews = len(positive_reviews)
    n_neg_reviews = len(negative_reviews)

    bags = []

    vocabulary = []
    
    freq_bigrams = []
    
    bigrams = []

    for review in positive_reviews:

        review_text = (review.find('title').string + review.find('review_text').string).lower()

        review_text = nltk.word_tokenize(review_text)

        #clear_text recebe argumentos
        if (clear):
            review_text = clear_text(review_text, stem, pos_t, lemm, rm_stop, rm_not_alpha)
        
        if (hand_neg):
            review_text = handle_negation(review_text)

        vocabulary.extend(review_text)
    
        bag = {}
               
        for word in review_text:
            if word in bag:
                bag[word] += 1
            else:
                bag[word] = 1
                
        bags.append(bag)
                 
          
    for review in negative_reviews:

        review_text = (review.find('title').string + review.find('review_text').string).lower()

        review_text = nltk.word_tokenize(review_text)

        
        if (clear_text):
            review_text = clear_text(review_text)
        
        review_text = handle_negation(review_text)
             
        vocabulary.extend(review_text)
        
        bag = {}

        for word in review_text:
            if word in bag:
                bag[word] += 1
            else:
                bag[word] = 1
        
        bags.append(bag)
        
    n_reviews = n_pos_reviews + n_neg_reviews
    
    vocabulary = list(set(vocabulary))
    
    matrix = np.zeros((n_reviews, len(vocabulary)), dtype="int")
      
    for i in range(n_reviews):
        for key in bags[i]:
            index = vocabulary.index(key)
            matrix[i][index] = bags[i][key]
                
    classes = np.zeros((n_pos_reviews + n_neg_reviews), dtype="int")
    classes[:n_pos_reviews] = 1
        
    return matrix, classes, vocabulary


def select_pd(X, Y, vocabulary, alpha=0.625):
    
    selected = []
    new_vocabulary = []
    
    positive = np.count_nonzero(X[Y == 1], axis=0)
    negative = np.count_nonzero(X[Y == 0], axis=0)
    
    pd_features = abs(positive - negative)/(positive + negative)
    
    for i in range(len(pd_features)):
        if pd_features[i] > alpha:
            selected.append(i)
            new_vocabulary.append(vocabulary[i])
        
    Xnew = X[:, selected]
    
    return Xnew, new_vocabulary

def select_df(X, vocabulary, alpha):
    
    selected = []
    new_vocabulary = []
    
    for i in range(X.shape[0]):
        nzero = (np.count_nonzero(X[:, i])/X.shape[0])
        if nzero > alpha:
            selected.append(i)
            new_vocabulary.append(vocabulary[i])
    
    Xnew = X[:, selected] 

    return Xnew, new_vocabulary

def select_c2(X, Y, vocabulary, alpha):
    
    selected = []
    new_vocabulary = []
    
    for i in range(len(vocabulary)):
        if chi_squared(X[:, i], Y, alpha):
            selected.append(i)
            new_vocabulary.append(vocabulary[i])
            
    Xnew = X[:, selected]
    
    return Xnew, new_vocabulary, selected

def chi_squared(X, Y, alpha):
    
    table = pd.crosstab(Y,X) 
    
    chi2, p, dof, expected = stats.chi2_contingency(table.values)
   
    if p < alpha:
        return True
    
    return False
    

def fp(X):
    
    return np.where(X > 0, 1, 0)


def tf_idf(X):
    
    m, n = X.shape 
    
    return X * np.log(m/np.count_nonzero(X, axis=0))
  
