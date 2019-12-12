# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:46:12 2019

@author: PG
"""

import pandas as pd
import preprocessor as p
#import string
#import re
import nltk
nltk.download('punkt')
#from nltk.tokenize import word_tokenize
#from nltk.tokenize import TreebankWordTokenizer
import csv
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

all_tweets = pd.read_csv(r'C:\Users\PG\Desktop\NLP final\alltweets.csv')
print (all_tweets)

tweets = all_tweets[['text']]

disc_rows = tweets.drop_duplicates()

disc_list = disc_rows['text'].values.tolist()


lowertext = [x.lower() for x in disc_list]



#https://stackoverflow.com/questions/34860982/replace-the-punctuation-with-whitespace


newlist = []
punctuations = '''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''

    
for line in lowertext:
    line = p.clean(line)
    line = ''.join([i for i in line if not i in punctuations])
    line = ''.join([i for i in line if not i.isdigit()])
    newlist.append(line)

#Because tweets by different usernames have different urls at the end of the tweet   
lowertext_disc = [] 
[lowertext_disc.append(x) for x in newlist if x not in lowertext_disc]


#https://codereview.stackexchange.com/questions/115954/writing-lists-to-csv-file
#https://gis.stackexchange.com/questions/72458/exporting-list-of-values-into-csv-or-txt-file-using-arcpy

with open("cleantext.csv", 'w') as write_outfile:
    writer = csv.writer(write_outfile, lineterminator='\n')
    for line in lowertext_disc:
        writer.writerow([line])
       

cleanText1 = pd.DataFrame(lowertext_disc, columns = ['tweets'])
cleanText1['tokens'] = ""

#def generate_ngrams(s, n):
    

#cleanText2 = cleanText1['tweets'].values.tolist()
#type(cleanText1['tokens'])
#results = []
#for line in cleanText2:
#    results.append(nltk.word_tokenize(cleanText2['tweets']))

#cleanText2 = cleanText1[:10]  
  
#tokenized_text = cleanText1['tweets'].apply(lambda x: x.split())
#cleanText1['tokens'] = tokenized_text
#print(tokenized_text)

#cleanText1['tweets'] = cleanText1['tweets']
#word_tokens = word_tokenize(cleanText1)
tokenized_text = cleanText1['tweets'].apply(lambda x: x.split())

wordcloud = WordCloud().generate(str(tokenized_text))

image = wordcloud.to_image()
image.show()
#cleanText2['tokens'] = tokenized_text
#tokenized_words = []
#for sentence in cleanText2['tweets']:
#    sentence_results = []
#    for s in sentence:
#        sentence_results.append(nltk.word_tokenize(sentence))
#    tokenized_words.append(sentence_results)


stop_words = set(stopwords.words('english'))

cleanText1['filtered-text'] = ""

#no_stopWords = []
no_stopWords = tokenized_text.apply(lambda x: [item for item in x if item not in stop_words])
##https://stackoverflow.com/questions/13464152/typeerror-unhashable-type-list-when-using-built-in-set-function/13464194
#new_tuple = tuple(tokenized_text)

removehyphen = ['–']
no_stopWords = no_stopWords.apply(lambda x: [item for item in x if item not in removehyphen])
cleanText1['filtered-text'] = no_stopWords

#https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f
#lemmatization

lemmatizer = WordNetLemmatizer()

def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text

lemmtext = cleanText1['filtered-text'].apply(lambda x:word_lemmatizer(x))

#Stemming
stemmer = PorterStemmer()

def word_stemmer(text):
    stem_text = " ". join([stemmer.stem(i) for i in text])
    return stem_text

stemtext = cleanText1['filtered-text'].apply(lambda x:word_stemmer(x))

wordcloud = WordCloud().generate(str(lemmtext))

image = wordcloud.to_image()
image.show()

#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.show()

#https://medium.com/fintechexplained/nlp-text-processing-in-data-science-projects-f083009d78fc

#https://stackoverflow.com/questions/48671270/use-sklearn-tfidfvectorizer-with-already-tokenized-inputs




t_train, t_val = train_test_split(lemmtext, test_size = 0.30)

def identity_tokenizer(text):
    return text

tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)    
vector1 = tfidf.fit_transform(t_train)
features = tfidf.get_feature_names()

#for line in t_train:
#    for word in line:
#        if word == 'abuse':
#            t_train.replace('Yes')
#        else:
#            t_train.replace('No')








