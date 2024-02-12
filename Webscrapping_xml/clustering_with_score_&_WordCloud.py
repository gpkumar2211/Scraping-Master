import os
import pandas as pd
import numpy as np
import bs4 as bs
import urllib.request
import re
import spacy
import re, string, unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lem=WordNetLemmatizer()

os.chdir(r"D:\Full Stack Data Science AI & ML\ClassNotes\Feb 5_Web scrapping on xml\xml_many articles")

from glob import glob
path=r"D:\Full Stack Data Science AI & ML\ClassNotes\Feb 5_Web scrapping on xml\xml_many articles"
all_files=glob(os.path.join(path,"*.xml"))

import xml.etree.ElementTree as ET

dfs=[]
for filename in all_files:
    tree=ET.parse(filename)
    root=tree.getroot()
    root=ET.tostring(root, encoding="utf8").decode('utf8')
    dfs.append(root)
    
dfs[0]

###########

import bs4 as bs
import urllib.request
import re

parsed_article=bs.BeautifulSoup(dfs[0],'xml')
paragraphs=parsed_article.find_all('p')

article_text_full=""
for p in paragraphs:
    article_text_full+=p.text
    print(p.text)
    
def data_preprocessing(each_file):
    parsed_article=bs.BeautifulSoup(each_file,'xml')
    paragraphs=parsed_article.find_all('para')
    
    article_text_full=''
    for p in paragraphs:
        article_text_full+=p.text
        print(p.text)
        
    return article_text_full

data=[data_preprocessing(each_file) for each_file in dfs]

##########################After Combine all articles#######################
from bs4 import BeautifulSoup
soup=BeautifulSoup(dfs[0],'html.parser')
print(soup.prettify())

parsed_article=bs.BeautifulSoup(dfs[0],'xml')
paragraphs=parsed_article.find_all('para')

def remove_stop_word(file):
    nlp=spacy.load('en_core_web_sm')
    punctuations=string.punctuation
    from nltk.corpus import stopwords
    stopwords=stopwords.words('english')
    SYMBOLS=" ".join(string.punctuation).split(" ")+["-", "...", "”","”"]
    stopwords=nltk.corpus.stopwords.words('english')+SYMBOLS
    
    doc=nlp(file, disable=['parser','ner'])
    tokens=[tok.lemma_.lower().strip() for tok in doc if tok.lemma_!='-PRON-']
    tokens=[tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    s=[lem.lemmatize(word) for word in tokens]
    tokens=' '.join(s)
    
    article_text=re.sub(r'\[[0-9]*\]',' ',tokens)
    article_text=re.sub(r'\s+',' ',article_text)
    
    formatted_article_text=re.sub('[^a-zA-Z]',' ',article_text)
    formatted_article_text=re.sub(r'\s+',' ',formatted_article_text)
    formatted_article_text=re.sub(r'\W*\b\w{1,3}\b',"",formatted_article_text)
    
    return formatted_article_text

clean_data=[remove_stop_word(file) for file in data]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

vectorizer=CountVectorizer(stop_words=stopwords.words('english')).fit(clean_data)

vectorizer.get_feature_names_out()
X=vectorizer.transform(clean_data).toarray()

data_final=pd.DataFrame(X, columns=vectorizer.get_feature_names_out())

from sklearn.feature_extraction.text import TfidfTransformer

tran=TfidfTransformer().fit(data_final.values)

X=tran.transform(X).toarray()

X=normalize(X)

from sklearn.cluster import KMeans
true_k=6
model=KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

cluster_result_data=pd.DataFrame(clean_data,columns=['text'])
cluster_result_data['group']=model.predict(X)

def token(sentence):
    tok=sentence.split()
    return tok

cluster_result_data['words']=[token(sentence) for sentence in cluster_result_data['text']]

def score(file):
    nlp=spacy.load('en_core_web_sm')
    punctuations=string.punctuation
    from nltk.corpus import stopwords
    stopwords=stopwords.words('english')
    SYMBOLS=" ".join(string.punctuation).split(" ")+["-", "...","”", "”" ]
    stopwords=nltk.corpus.stopwords.words('english')+SYMBOLS
    
    doc=nlp(file, disable=['parser','ner'])
    tokens=[tok.lemma_.lower().strip() for tok in doc if tok.lemma_!='-PRON-']
    tokens=[tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    s=[lem.lemmatize(word) for word in tokens]
    tokens=' '.join(s)
    
    article_text=re.sub(r'[[0-9]*\]',' ',tokens)
    article_text=re.sub(r'\s+',' ',article_text)
    
    formatted_article_text=re.sub('[^a-zA-Z]',' ',article_text)
    formatted_article_text=re.sub(r'\s+',' ',formatted_article_text)
    formatted_article_text=re.sub(r'\W*\b\w{1,3}\b',"",formatted_article_text)
    
    word_frequencies={}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word]=1
            else:
                word_frequencies[word]+=1
                
    maximum_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=(word_frequencies[word]/maximum_frequency)
        
    nlp=spacy.load("en_core_web_sm")
    doc=nlp(file)
    sentence_list=[str(sentence) for idno, sentence in enumerate(doc.sents)]
    
    sentence_scores={}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent]=word_frequencies[word]
                else:
                    sentence_scores[sent]+=word_frequencies[word]
                    
    mean_value=np.mean(list(sentence_scores.values()))
    return mean_value

cluster_result_data['score']=[score(sen) for sen in data]
                    
#own try for cloud images
#create a word for each cluster

from wordcloud import  WordCloud
import matplotlib.pyplot as plt

grouped_data=cluster_result_data.groupby('group')['text']

#generate word cloud for each cluster
for group_name, group_text in grouped_data:
    combined_text=" ".join(group_text.tolist())
    
    #create wordcloud with customizations
    wordcloud=WordCloud(background_color='black',width=1200, height=800).generate(combined_text)
    
    #plot the word cloud
    plt.figure(figsize=(8,6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(f"Cluster {group_name + 1}")
    plt.axis('off')
    plt.show()
    

