import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 

def bag_of_words(category, option):
       
    category += '/'

    positive = open('sorted_data_acl/' + category  + 'positive.review', 'r')
    negative = open('sorted_data_acl/' + category  + 'negative.review', 'r')

    positive_reviews = (BeautifulSoup(positive, 'lxml'))
    negative_reviews = (BeautifulSoup(negative, 'lxml'))

    positive_reviews = positive_reviews.find_all(['review_text', 'title']) 
    negative_reviews = negative_reviews.find_all(['review_text', 'title']) 

    positive_words = []
    negative_words = []
    
    for sentence in positive_reviews:
        positive_words.extend(nltk.word_tokenize(sentence.string.lower()))
    
    for sentence in negative_reviews:
        negative_words.extend(nltk.word_tokenize(sentence.string.lower()))
   
    bag_positive = {}
    bag_negative = {}
    
    
    stop_words = stopwords.words('english')
   
    lemmatizer = WordNetLemmatizer() 
    
    for word, pos in nltk.pos_tag(positive_words):
        if (option and (pos[0] == 'V' or pos[0] == 'J' or pos[0] == 'R')) or not option:
            if word.isalpha() and word not in stop_words:
                    w = lemmatizer.lemmatize(word)   
                    if w in bag_positive: 
                        bag_positive[w] += 1
                    else:
                        bag_positive[w] = 1
     
                        
    for word, pos in nltk.pos_tag(negative_words):
        if (option and (pos[0] == 'V' or pos[0] == 'J' or pos[0] == 'R')) or not option:
            if word.isalpha() and word not in stop_words:
                    w = lemmatizer.lemmatize(word)   
                    if w in bag_negative: 
                        bag_negative[w] += 1
                    else:
                        bag_negative[w] = 1

                    
    df = pd.DataFrame(data=[bag_negative, bag_positive])

    df.fillna(0, inplace=True)

    return df

def test_bag(words, category, option):
    
    category += '/'

    unlabeled = open('sorted_data_acl/' + category  + 'unlabeled.review', 'r')

    unlabeled_reviews = (BeautifulSoup(unlabeled, 'lxml'))
    
    unlabeled_reviews = unlabeled_reviews.find_all(['review_text', 'title']) 

    unlabeled_words = []
    
    for sentence in unlabeled_reviews:
        unlabeled_words.extend(nltk.word_tokenize(sentence.string.lower()))
    
    bag_unlabeled = {}
    
    for word in words:
        bag_unlabeled[word] = 0
    
    stop_words = stopwords.words('english')
   
    lemmatizer = WordNetLemmatizer() 
    
    
    for word, pos in nltk.pos_tag(unlabeled_words):
        if (option and (pos[0] == 'V' or pos[0] == 'J' or pos[0] == 'R')) or not option:
            if word.isalpha() and word not in stop_words:
                    w = lemmatizer.lemmatize(word)   
                    if w in words:    
                        if w in bag_unlabeled: 
                            bag_unlabeled[w] += 1
                        else:
                            bag_unlabeled[w] = 1

    df = pd.DataFrame(data=[bag_unlabeled])
    
    x = df.iloc[:, :].values 
    
    return x