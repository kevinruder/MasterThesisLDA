import requests
import logging
from gensim import corpora,models, similarities
from bs4 import BeautifulSoup
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict
import pprint
from stop_words import get_stop_words
import csv
import string

documents_image_A = []
documents_image_B = []
documents_image_C = []
documents_image_D = []
documents_image_E = []
documents_image_F = []
documents_image_G = []
documents_image_H = []
documents_image_I = []
documents_image_J = []


stop_words = get_stop_words('english')

csv_location = "C:/Users/Kevin/Documents/Semester Project/MASTERS/Data/QUESTIONAIRE RESULTS.csv"


def extract_document_from_csv(csv_location):

    source = open(csv_location, "rt")
    rdr = csv.reader(source)
    for row in rdr:

        global documents_image_A
        global documents_image_B
        global documents_image_C
        global documents_image_D
        global documents_image_E
        global documents_image_F
        global documents_image_G
        global documents_image_H
        global documents_image_I
        global documents_image_J

        if(row[1] == "A"):

            documents_image_A.append(row[2])
            documents_image_B.append(row[3])
            documents_image_C.append(row[4])
            documents_image_D.append(row[5])
            documents_image_E.append(row[6])


        else:

            documents_image_F.append(row[2])
            documents_image_G.append(row[3])
            documents_image_H.append(row[4])
            documents_image_I.append(row[5])
            documents_image_J.append(row[6])





def extract_tfidf_index(corpus_tfidf):

    #extracts all indexes/words with a TFIDF of above 0.5

    indexes = []

    for doc in corpus_tfidf:
        for a in doc:
            #0.5 is the TFIDF THRESHOLD AND IS HARDCODED ATM, SHOULD BE A VARIABLE THAT IS DECLARED ON TOP of code
            if (a[1] > 0.5):
                indexes.append(a[0])

    return indexes

def find_word_with_index(list_of_index,word_identifier_list):

    #looks at the list of indexes and prints out the word that is connected to the Identifier

    for doc in word_identifier_list:
        for index in list_of_index:
            if word_identifier_list[doc] == index:
                print(doc)

extract_document_from_csv(csv_location)

print(documents_image_A)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

stoplist = set(stop_words)

print(stoplist)

texts = [[word for word in document.lower().split() if word not in stoplist]for document in documents_image_H]

frequency = defaultdict(int)


for text in texts:
    for token in text:
        #remove punctutation
        token = token.translate(str.maketrans('', '', string.punctuation))
        print(token)
        #count frequency
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]for text in texts]

print(texts)

dictionary = corpora.Dictionary(texts)

dictionary.save('temporary.dict')

word_identifier_list = dictionary.token2id

corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize('temporary.mm',corpus)

#for poop in corpus:
   #print(poop)

tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]

list_of_index = extract_tfidf_index(corpus_tfidf)

find_word_with_index(list_of_index,word_identifier_list)

#Have to swap between keys and values in dict or else LDAmodel Will not recognize the format
new_dict = {y:x for x,y in word_identifier_list.items()}

for doc in corpus_tfidf:
    print(doc)

lsi = models.LdaModel(corpus = corpus_tfidf, num_topics= 3 ,id2word=new_dict)


# I tried different number of topics, and with too many topics, the results were very similar
# Assuming there will maximum be 3 different stories, the number of topics is set to 3


lsi.print_topics(10)

print(lsi.show_topic(1))

