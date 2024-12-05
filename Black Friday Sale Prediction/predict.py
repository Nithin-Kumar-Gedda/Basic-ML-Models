import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')
# print(df.head())
# print(df.describe())
# print(df.info())
# print(df.isnull().sum())
# print(df.apply(lambda x: len(x.unique())))

plt.style.use("fivethirtyeight")

# occ_plot = df.pivot_table(index='Occupation', values='Purchase', aggfunc=np.mean)
# occ_plot.plot(kind='bar')
# plt.xlabel('Occupation')
# plt.ylabel('Purchase')
# plt.title('Occupation and Purchase analysis')
# plt.legend()
# plt.show()

df['Product_Category_2']= df['Product_Category_2'].fillna(-2.0).astype('float32')
df['Product_Category_3']= df['Product_Category_3'].fillna(-2.0).astype('float32')

# Encoding
gender_dict = {'F':0, 'M':1}
df['Gender'] = df['Gender'].apply(lambda x: gender_dict[x])

cols=['Age', 'City_Category', 'Stay_In_Current_City_Years']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

# print(df.head())

# Cor = df.corr()
# sns.heatmap(Cor, annot=True, cmap='coolwarm')
# plt.show()
user_id = df['User_ID']
df.drop(columns=['User_ID', 'Product_ID'], inplace=True)
# print(df.head())
x = df.drop(columns=['Purchase'])
y = df['Purchase']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


def train(model, x, y):
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cv_score = cross_val_score(model, x, y, scoring='neg_mean_squared_error')
    print("CV Scores:", cv_score)
    
    print("MSE:",mean_squared_error(y_pred,y_test))
    print('Avg CV_score:',np.mean(cv_score))


# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
train(model, x, y)


pred = model.predict(x_test)
submission = pd.DataFrame()
submission['User_ID'] = user_id.loc[x_test.index]
submission['Purchase'] = pred

submission.to_csv('submission.csv', index=False)