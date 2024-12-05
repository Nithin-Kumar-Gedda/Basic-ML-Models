import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import re
import string
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

#Load dataset

df=pd.read_csv('Twitter Sentiments.csv')
# print(df.head())
# print(df.describe())
# print(df.info())

#Preprocessing dataset

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word,"",input_txt)
    return input_txt

#Removing twitter handles (@user)
df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
# df['clean_tweet'] = df['tweet'].apply(lambda x: remove_pattern(x, "@[\w]*"))

# remove special characters, numers and punctuations
df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")

#remove short words
df['clean_tweet'] = df['clean_tweet'].apply(lambda x:" ".join([w for w in x.split() if len(w)>3]))

#Tokenization
tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())

#Stemming
stemmer=PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])

df['clean_tweet'] = tokenized_tweet.apply(lambda sentence: " ".join([word for word in sentence]))
# print(sentence_tweet.head())
# print(df.head())

# Wordcloud for all words 

# all_words = " ".join([sentence for sentence in df['clean_tweet']])
# wordcloud = WordCloud(width=800,height=500, random_state=42,max_font_size=100).generate(all_words)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# generate the wordcloud for negative words
all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label']==1]])

wordcloud = WordCloud(width=800,height=500, random_state=42,max_font_size=100).generate(all_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
# plt.show()

bow_vectorizer = CountVectorizer(max_df=0.90,min_df=2,max_features=1000,stop_words='english')
bow = bow_vectorizer.fit_transform(df['clean_tweet'])

x_train, x_test, y_train, y_test = train_test_split(bow, df['label'],test_size=0.25,random_state=42)

# model = LogisticRegression()
# model= DecisionTreeClassifier()
model=RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("F1 score:",f1_score(y_test, y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))