import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier


warnings.filterwarnings('ignore')

df = pd.read_csv('winequalityN.csv')
# print(df.head())
# print(df.info())

# print(df.isnull().sum())
for col, value in df.items():
    if col != 'type':
        df[col]=df[col].fillna(df[col].mean())
# print(df.isnull().sum())

# fig , ax = plt.subplots(ncols=6, nrows=2, figsize=(15,10))
index = 0
# ax = ax.flatten()
for col, value in df.items():
    if col != 'type':
        # sns.boxplot(value, ax= ax[index])
        index +=1
# plt.tight_layout(pad = 0.5,w_pad=0.7, h_pad=5.0)
# plt.show()

# Log transformation....
df['volatile acidity'] = np.log(df['volatile acidity']+1)
df['citric acid'] = np.log(df['citric acid']+1)
df['residual sugar'] = np.log(df['residual sugar']+1)
df['chlorides'] = np.log(df['chlorides']+1)
df['free sulfur dioxide'] = np.log(df['free sulfur dioxide']+1)
df['density'] = np.log(df['density']+1)

# sns.countplot(x=df['type'])
# plt.show()

# Co-relation metrics
df_temp = df.drop(columns=['type'], axis=1)
cor = df_temp.corr()
# sns.heatmap(cor, annot=True,cmap='coolwarm')
# plt.show()

x = df.drop(columns=['type', 'quality','density'], axis=1)
y = df['quality']


# Class imbalancement
sample = SMOTE(k_neighbors=4)
x, y = sample.fit_resample(x,y)
# print(y.value_counts())

def train(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model.fit(x_train,y_train)
    print("Accuracy:", model.score(x_test,y_test))
    cv_score = cross_val_score(model,x,y)
    print("CV_Score:",np.mean(cv_score))


# model = LogisticRegression()
# model = DecisionTreeClassifier()
model = ExtraTreesClassifier()
train(model,x,y)