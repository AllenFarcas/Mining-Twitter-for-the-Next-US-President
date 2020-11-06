import pandas as pd
import numpy as np
import re
import nltk

#Data preprocessing to removes @usernames,urls,symbols and makes all text lowercase
def preprocess_text(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', text)
    text = re.sub('@[^\s]+','', text)
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+',' ', text)
    text = re.sub(' +',' ', text)
    return text.strip()

if __name__ == "__main__":	
	#load data from csv/excel/json
	data=pd.read_excel('tweets.xlsx') 
	#data.head()

	text = [preprocess_text(t) for t in data.iloc[:]['full_text']]

	#storing preprocessed texts in a separate column 
	data["processed_text"]= text

	#example
	print(data.iloc[0]['full_text'])
	print(data.iloc[0]["processed_text"])

	#data.head()

	#Tokenizing 
	tokens = [nltk.word_tokenize(t) for t in data.iloc[:]['processed_text']]
	#print(tokens)





