import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor

warnings.filterwarnings('ignore')

df = pd.read_csv("Boston Dataset.csv", encoding='utf-8')
# print(df.head())
df.drop(columns=['Unnamed: 0'], inplace=True, axis=0)
# print(df.info())
# print(df.isnull().sum())

# fig, axis = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
index =0
# axis = axis.flatten()
for col, value in df.items():
    # sns.boxplot(y=col, data=df, ax=axis[index])
    index +=1
# plt.tight_layout()
# plt.show()

# Min Max Normalization

cols = ['crim', 'zn', 'tax', 'black']
for col in cols:
    mini = min(df[col])
    maxi = max(df[col])
    df[col] = (df[col]-mini) / (maxi - mini)

# Standardization

scaler = StandardScaler()
scaled_cols = scaler.fit_transform(df[cols])
scaled_cols = pd.DataFrame(scaled_cols, columns=cols)
# print(scaled_cols.head())
for col in cols:
    df[col] = scaled_cols[col]

# Co-relation metrics

corr = df.corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.show()

x = df.drop(columns=['rad', 'medv'], axis=1, inplace=True)
y = df['medv']

def train(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)
    model.fit(x_train,y_train)

    pred = model.predict(x_test)

    cv_score = cross_val_score(model, x, y, scoring='neg_mean_squared_error')
    cv_score = np.abs(np.mean(cv_score))

    print("model report")
    print("MSE:",mean_squared_error(y_test, pred))
    print("Cv_score:",cv_score)


model = LinearRegression(normalize = True)
train(model, x, y)