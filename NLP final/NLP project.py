# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:11:14 2019

@author: PG
"""

import pandas as pd
#from io import StringIO
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
#import csv
from nltk.tokenize import TreebankWordTokenizer
import re #regular expression
#import string
import preprocessor as p
#from itertools import chain

all_tweets = pd.read_csv(r'C:\Users\PG\Desktop\NLP final\alltweets.csv')
print (all_tweets)

tweets = all_tweets[['text']]

disc_rows = tweets.drop_duplicates()

#for elem in disc_rows:
#    elem = elem.lower()

#lowertext = list(map(lambda text:text.lower(), disc_rows))

lowertext = [x.lower() for x in disc_rows]

disc_list = disc_rows.values.tolist()

lowertext1 = [x.lower() for x in disc_list]



#def lower_list(disc_list):
#    strings=[]
#    for line in disc_list:
#        strings.append(string.lower())
#    return strings

def lower_text(x):
    print(list(x.lower()))

map(lower_text, disc_list)
#    x = x.lower()
#    lowertext = map(lambda x:x.lower(), disc_list)



print(lowertext)

for item in disc_list:
    item.lower()

type(lowertext)


for tweet in lowertext:
    print(tweet)

lowertext1 = lowertext.to 

for tweet in lowertext:
    print(tweet)

clean_text = p.clean(lowertext)




#text = all_tweets[['text']]

#def cleantext(tweetlist1):
for tweet in tweetlist1:
    no_num = p.clean(tweet)
    print(tweet)

#for line in range(disc_rows):
#    [@a-z].*, ""
    

#for line in range(disc_rows):
#    urls = (e["url"] for e in line["urls"])
#    users = ("@"+e["screen_name"] for e in line["user_mentions"])
#    text = reduce(lambda t,s: t.replace(s, ""), chain(urls, users), text)
#    result.append(text)


#type(text)
#text_csv = disc_rows.to_csv(r'C:\Users\PG\Desktop\NLP final\textcolumn.csv', index=False)
#
#no_index = disc_rows.to_string(index=False)


clean_text = p.clean(no_index)

clean_text = re.sub('[^a-zA-Z\s]', '', clean_text)

#no_numbers = no_numbers.replace("\r" + "\n", " ")


no_punc = no_numbers.translate(str.maketrans("","", string.punctuation))





#lowercase_csv = lowercase.to_csv(r'C:\Users\PG\Desktop\NLP final\lowercase.csv', index=False)
#lowercase_df = pd.read_csv(r'C:\Users\PG\Desktop\NLP final\lowercase.csv')



#list = sent_tokenize(lower_case)
#print(list)


tokenizer = TreebankWordTokenizer()
print(word_tokenize(no_punc))