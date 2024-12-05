import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


warnings.filterwarnings('ignore')

train = pd.read_csv("Train.csv")
# print(df.head())
# print(df.isnull().sum())
# print(df.apply(lambda x: len(x.unique())))  calculates the unique values in the dataset col....
test = pd.read_csv('Test.csv')
df = pd.concat([train, test], axis=0)
df.reset_index(drop=True, inplace=True)
train_len = len(train)
test_len = len(test)

#  for categorical columns
cat_col = []
for i in train.dtypes.index:
    if train.dtypes[i] == 'object':
        cat_col.append(i)
# print(cat_col)
cat_col.remove('Item_Identifier')
cat_col.remove('Outlet_Identifier')
# print(cat_col)
# for col in cat_col:
    # print(col)
    # print(df[col].value_counts())
    # print()
# print(df['Item_Weight'].isnull().sum())
df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
# print(df.isnull().sum())
df['Outlet_Size'].fillna(df['Outlet_Size'].mode(), inplace= True)
# print(sum(df['Item_Visibility']==0))
df.loc[:, 'Item_Visibility'].replace([0],[df['Item_Visibility'].mean()], inplace=True)
# print(df['Item_Fat_Content'].value_counts())
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
df['New_Item_Type'] = df['Item_Identifier'].apply( lambda x : x[:2])
df['New_Item_Type'] = df['New_Item_Type'].map({'FD': 'Food', 'DR': 'Drink', 'NC':'Non-Consumable'})
df.loc[df['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
df ['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']

# print(df.head())
# sns.displot(df['Item_Visibility'])
df['Item_Outlet_Sales'] =np.log(1+df['Item_Outlet_Sales'])
# sns.displot(df['Item_Outlet_Sales'])

l = list(df['Item_Type'].unique())
# chart = sns.countplot(x=df['Item_Type'])
# chart.set_xticklabels(labels=l, rotation = 90)

le = LabelEncoder()

df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
for col in cat_col:
    df[col] = le.fit_transform(df[col])
# plt.show()
# df.drop(columns=['Item_Identifier'])
# cor = df.corr()
# sns.heatmap(cor, annot=True, cmap='coolwarm')
# plt.show()

df['Outlet_Size'] = le.fit_transform(df['Outlet_Size'])
df['Outlet_Location_Type'] = le.fit_transform(df['Outlet_Location_Type'])
df['Outlet_Type'] = le.fit_transform(df['Outlet_Type'])

train = df.iloc[:train_len,:]
test = df.iloc[train_len:,:]
test = test.drop(columns=['Item_Outlet_Sales'])


x =train.drop(columns=['Item_Identifier', 'Outlet_Establishment_Year', 'Item_Outlet_Sales', 'Outlet_Identifier'])
y = train['Item_Outlet_Sales']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state= 42)

def train(model, x, y):
    model.fit(x,y)
    pred = model.predict(x)

    cv_score = cross_val_score(model, x, y, scoring = 'neg_mean_squared_error')
    cv_score = np.abs(np.mean(cv_score))

    print('Model Report')
    print("MSE:",mean_squared_error(y,pred))
    print("CV Score:", cv_score)

# model = LinearRegression()
# model = Lasso()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = Ridge()
# model = ExtraTreesRegressor()
# train(model, x, y)

# Prediction on test data

x_test=test.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier'])
# print(x_test.head())
model = RandomForestRegressor()
model.fit(x, y)
pred = model.predict(x_test)
# print(pred)
submission =pd.DataFrame()
submission['id'] = test['Item_Identifier']
submission['sales'] = pred
submission.to_csv('test_result.csv', index=False)

# Prediction on real time data

# result=model.predict([x_test.iloc[0]]) # in place of it you have to pass the all values in column-wise 
result = model.predict([[20.750000,0.000000,0.007565,13.000000,107.862200,1.000000,0.000000,1.000000,1.000000,14.000000,9.000000]]) 
print(result)