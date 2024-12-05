import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Models

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

df = pd.read_csv('HeartDisease.csv') # load the dataset

# print(df.head()) # To print the first five lines...
# print(df.info()) # To check the data types.
# print(df.desc()) # discriptive statistics of DataFrame.
# print(df.isnull().sum()) # Check the null values in the each column.

# Exploratory Data Analysis - EDA 

# sns.countplot(x=df['target'])  
# sns.countplot(x=df['gender'])

scaler = StandardScaler()

temp = df.drop(columns=['age', 'gender', 'target'], axis=1)
# fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(20, 10))
# index=0
# ax = ax.flatten()
# for col in temp.columns:
#     if index >= len(ax):  # Check if index exceeds the number of subplots
#         break
    # sns.distplot(temp[col], ax=ax[index])
    # index +=1
# plt.tight_layout(pad=0.5, h_pad=5, w_pad=0.5)
# plt.show()

  # Correlation metrics

cor = df.corr()
# sns.heatmap(cor, annot=True, cmap='coolwarm')
# plt.show()

x = df.drop(columns=['age','gender','target'],axis=1)
y = df['target']

x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# model = LogisticRegression()  # Logistic Regression model

# model = DecisionTreeClassifier() # Decision Tree Model

model = RandomForestClassifier()  # Random Forest model

model.fit(x_train, y_train)

y_pred = model.predict(x_test) 

print(classification_report(y_test, y_pred)) # Classification Report

print("Accuracy :",accuracy_score(y_pred,y_test)*100) # Accuracy of the model




