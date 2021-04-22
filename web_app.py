# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud,STOPWORDS
from PIL import Image
import string

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from pickle import dump, load

dataset_loc = "SMSSpamCollection"
img_location = "data/spam_img.jpg"

# sidebar
def load_sidebar():
    st.sidebar.subheader("SMS Spam Collection")
    st.sidebar.success("Contains one set of SMS messages in English of 5,572 messages, tagged according being ham (legitimate) or spam.")
    st.sidebar.info("This data originally came from SMS's Data for Everyone library.")
    st.sidebar.warning("Made by Pooja Bhather :heart:")

@st.cache
def load_data(dataset_loc):
    df = pd.read_csv(dataset_loc, sep="\t", names = ['Category','Text'])
    return df


def load_description(df):
        # Preview of the dataset
        st.header("Data Preview")
        preview = st.radio("Choose Head/Tail?", ("Top", "Bottom"))
        if(preview == "Top"):
            st.write(df.head())
        if(preview == "Bottom"):
            st.write(df.tail())

        # display the whole dataset
        if(st.checkbox("Show complete Dataset")):
            st.write(df)
            
        if(st.checkbox("Show data description")):
            st.write(df.describe(include = "all"))

        # Show shape
        if(st.checkbox("Display the shape")):
            st.write(df.shape)
            dim = st.radio("Rows/Columns?", ("Rows", "Columns"))
            if(dim == "Rows"):
                st.write("Number of Rows", df.shape[0])
            if(dim == "Columns"):
                st.write("Number of Columns", df.shape[1])

        # show columns
        if(st.checkbox("Show the Columns")):
            st.write(df.columns)
            if(st.checkbox("Show the numbers of ham and spam")):
                st.write(df['Category'].value_counts())
            
def count_words(df):
    stopwords=STOPWORDS
    mapping = dict.fromkeys(map(ord,string.punctuation))
    words = {}
    for m in df['Text']:
        try:
            sent = m.split(' ')
        except:
            pass
        for word in sent:
            word = word.strip()
            word = word.lower() # convert into lowercase
            word = word.translate(mapping) # remove all punctuations
            if word in stopwords:  # remove all stopwords
                continue
            if len(word) < 3 or len(word) > 9: # remove unneccessary words
                continue
            if word in words:
                words[word]+=1
            else:
                words[word]=1
    return words
    


# WordCloud

def load_wordcloud(df, kind):
    
    df_temp = df.loc[df['Category']== kind, :]
    
    words = ' '.join(df_temp['Text'])
    cleaned_word = " ".join([word for word in words.split() if len(word)>3]) 
    
    mask = np.array(Image.open("twitter_mask.png"))
    wc = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      colormap='Spectral',
                      mode='RGBA',
                      width=700,
                      height=400,
                      mask=mask
                     ).generate(cleaned_word)
    wc.to_file("data/wc.png")
    st.image("data/wc.png", use_column_width = True)
    
    st.subheader("Most Frequent Words of "+kind)
    words = count_words(df_temp)
    word_df = pd.DataFrame(words.items(),columns=['words','count']) # list to dataframe
    word_df.sort_values('count',ascending=False,inplace=True) # sort according to count
    word_df[word_df['count']>100].plot(x='words',kind='bar')  # plot bar graph of important words
    st.pyplot()
    

    
def load_viz(df):
        st.header("Data Visualisation")
        # show tweet sentiment count
        st.subheader("Seaborn - Ham/Spam Count")
        st.write(sns.countplot(x='Category', data=df))
        st.pyplot()

        # ***************
        st.subheader("Plotly - Ham/Spam Count")
        temp = pd.DataFrame(df['Category'].value_counts())
        fig = px.bar(temp, x=temp.index, y='Category')
        st.plotly_chart(fig, use_container_width=True)
        # ***************
        
        # Show Donut Chart
        st.subheader("Donut Chart")
        df.Category.value_counts().plot(kind='pie', labels=['ham', 'spam'],
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                              textprops={'fontsize': 15})
        st.pyplot()
        
        # Show WordCloud
        st.subheader("Word Cloud")
        type = st.radio("Choose the sentiment?", ("ham", "spam"))
        load_wordcloud(df, type)
        
def preprocess(mssg):
    # Removing special characters and digits
    letters_only = re.sub("[^a-zA-Z]", " ",mssg)
    
    # change sentence to lower case
    letters_only = letters_only.lower()

    # tokenize into words
    words = letters_only.split()
    
    # remove stop words                
    words = [w for w in words if not w in stopwords.words("english")]
    
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    clean_sent = " ".join(words)
    
    return clean_sent

def predict(mssg):
    
    # Loading pretrained CountVectorizer from pickle file
    vectorizer = load(open('pickle/countvectorizer.pkl', 'rb'))
    
    # Loading pretrained logistic classifier from pickle file
    classifier = load(open('pickle/logit_model.pkl', 'rb'))
    
    # Preprocessing the tweet
    clean_text = preprocess(mssg)
    
    # Converting text to numerical vector
    clean_text_encoded = vectorizer.transform([clean_text])
    
    # Converting sparse matrix to dense matrix
    text_input = clean_text_encoded.toarray()
    
    # Prediction
    prediction = classifier.predict(text_input)
    
    return prediction
        

def main():

    # sidebar
    load_sidebar()

    # Title/ text
    st.title('SMS Spam Collection')
    st.image("data/email-bomb.jpg", use_column_width = True)
    st.text('Analyze the message is spam or not')

    # loading the data
    df = load_data(dataset_loc)

    # display description
    load_description(df)

    # data viz
    load_viz(df)
    st.title("Message Prediction")
    mssg = st.text_input('Enter your message')

    prediction = predict(mssg)

    if(mssg):
        st.subheader("Prediction:")
        if(prediction == 0):
            st.write("Spam Message :cry:")
        else:
            st.write("Ham Message :sunglasses:")


if(__name__ == '__main__'):
    main()