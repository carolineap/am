import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
import scipy.stats as stats
from scipy.stats import chi2_contingency
from nltk import FreqDist

   
def remove_not_alpha(words):
    
    only_alpha = [word.replace("'", "o") for word in words if word.isalpha() or (word == "n't")]
    
    return only_alpha
    
def remove_stop_words(words):
    
    stop_words = stopwords.words('english')
    
    without_sw = [word for word in words if word not in stop_words]
    
    return without_sw

def stemming(words):
    
    stemmer = PorterStemmer() 
   
    stemming_words = [ stemmer.stem(word) for word in words]
    
    return stemming_words
    
def get_frequency(words):
    
    bag = {}
    
    for word in words:
        if word in bag:
            bag[word] += 1
        else:
            bag[word] = 1
    
    return bag

def clear_text(words):
    return stemming(remove_stop_words(remove_not_alpha(words)))
    
def chi_squared(X, Y, alpha=0.05):
    
    table = pd.crosstab(Y,X) 
    
    chi2, p, dof, expected = stats.chi2_contingency(table.values)
   
    if p < alpha:
        return True
    
    return False

def bow(category):
           
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

    for review in positive_reviews:

        review_text = (review.find('title').string + review.find('review_text').string).lower()

        review_text = nltk.word_tokenize(review_text)

        review_text = clear_text(review_text)

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

        review_text = clear_text(review_text)

        bag = {}

        for word in review_text:
            if word in bag:
                bag[word] += 1
            else:
                bag[word] = 1

        bags.append(bag)

    df = pd.DataFrame(data=bags) 
    #df.dropna(axis=1, thresh=10, inplace=True)
    df.fillna(0, inplace=True)

    list = []
    classes = np.zeros((n_pos_reviews + n_neg_reviews), dtype="int")
    classes[:n_pos_reviews] = 1
    for column in df.columns:
        if chi_squared(df[column].values, classes):
            list.append(column)
        
    df_new = df[list]  
    
    return df, df_new, classes


def stratified_holdOut(target, pTrain):
    
    train_index = []
    test_index = []
    
    classes = np.unique(target)
    i = []
    for c in classes:
        i, = np.where((target == c))
        p = round(pTrain*len(i))
        train_index.extend(i[:p])
        test_index.extend(i[p:])    
    
    train_index.sort()
    test_index.sort()    
    
    return train_index, test_index