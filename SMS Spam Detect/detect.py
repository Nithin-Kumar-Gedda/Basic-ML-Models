import pandas as pd 
import numpy as np 
import nltk
nltk.download('punkt')
import re
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
# print(df.head())
df = df.drop(columns=['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], axis=1)
# print(df.head())
df = df.rename(columns={'v2': 'messages','v1': 'label'})

# print(df.isnull().sum())

StopWords = set(stopwords.words('english'))
def clean_text(text):
    text=text.lower()
    text = re.sub(r'^[0-9a-zA-Z]',' ', text) # it will remove the special characters
    text = re.sub(r'\s+',' ', text) # it will remove extra spaces
    text = " ".join(word for word in text.split() if word not in StopWords) # it will remove stopwords
    return text

df['clean_text'] = df['messages'].apply(clean_text)
# print(df.head())

x = df['clean_text']
y = df['label']

vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(x)

def classify(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model.fit(x_train,y_train)

    print('Acc:',model.score(x_test,y_test))
    cv_score = cross_val_score(model, x, y)
    print('CV Score:',np.mean(cv_score))

model = LogisticRegression()
classify(model,X_transformed,y)
