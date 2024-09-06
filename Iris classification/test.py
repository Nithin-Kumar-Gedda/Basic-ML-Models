import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import warnings

warnings.filterwarnings('ignore')


df = pd.read_csv('Iris.csv')

# print(df.describe())
# print(df.head())

df = df.drop(columns=['Id'])   #delete a column

# print(df.info())
# print(df['Species'].value_counts()) #print the count of sample on each item
# print(df.isnull().sum()) # it print the sum of null values if it is in the dataset.


# Exploratory data analysis

# Histograms

# df['SepalLengthCm'].hist()
# df['SepalWidthCm'].hist()
# df['PetalLengthCm'].hist()
# df['PetalLengthCm'].hist()
# plt.show()

#ScatterPlots

colors = ['red', 'blue', 'green']
species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c=colors[i], label = species[i])

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.legend()
# plt.show()

# label encoder , it is used to convert the string format to numeric form... which can be understood by machine.

# le = LabelEncoder()
# df['Species'] = le.fit_transform(df['Species'])

X = df.drop(columns=['Species'])
Y = df['Species']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3)

#Logistic Regression

# model = LogisticRegression()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)

#KNN

# model = KNeighborsClassifier()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)

#Decision Tree

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# print('Actual values:', y_test.values)
# print('Predicted values:', y_pred)
# # Accuracy
# print(f"Accuracy:{model.score(x_test,y_test)}")


filename = "savemodel.sav"
pickle.dump(model, open(filename, 'wb'))
load_model = pickle.load(open(filename, 'rb'))

result = load_model.predict([[6.0, 2.2, 4.0, 1.0]])
print(result)