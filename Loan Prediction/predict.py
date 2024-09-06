import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix



df = pd.read_csv("train.csv")
# print(df.head())
# print(df.info())
# print(df.isnull().sum())
df = df.drop(columns=['Loan_ID'])

#   Data Analysis

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
# print(df.isnull().sum())

#   Exploratory Data Analysis

# sns.countplot(df['Gender'])
# sns.countplot(x ='Married', data=df)
# sns.countplot(x = df['Dependents'])

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome']+1)
# sns.displot(df['ApplicantIncomeLog'])

df['LoanAmountLog'] = np.log(df['LoanAmount']+1)
# sns.displot(df['LoanAmountLog'])

df['LoanAmountTermLog'] = np.log(df['Loan_Amount_Term']+1)
# sns.displot(df['LoanAmountTermLog'])

df['CoapplicantIncomeLog'] = np.log(df['CoapplicantIncome']+1)
# sns.displot(df['CoapplicantIncomeLog'])

df['TotalIncomeLog'] = np.log(df['TotalIncome']+1)
# sns.displot(df['TotalIncomeLog'])

# plt.show()
# print(df.head())
# print(df.dtypes)

# Corelation #Label Encoding
labelencoder = LabelEncoder()

df['Gender'] = labelencoder.fit_transform(df['Gender'])
df['Married'] = labelencoder.fit_transform(df['Married'])
df['Education'] = labelencoder.fit_transform(df['Education'])
df['Self_Employed'] = labelencoder.fit_transform(df['Self_Employed'])
df['Property_Area'] = labelencoder.fit_transform(df['Property_Area'])
df['Dependents'] = labelencoder.fit_transform(df['Dependents'])
df['Loan_Status'] = labelencoder.fit_transform(df['Loan_Status'])

cor = df.corr()
# sns.heatmap(cor, annot=True, cmap="BuPu" )
# plt.show()

cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'TotalIncome', 'CoapplicantIncomeLog']
df = df.drop(columns=cols, axis=1)
# print(df.head())

# Train-Test Split

x = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']

x_trian, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Model Training 
def classify(model, x, y):
    x_trian, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model.fit(x_trian, y_train)
    print("Accuracy is :", model.score(x_test,y_test)*100)

    # Cross validation - it is used for better validation of model
    # eg : cv-5, train - 4, test -1
    score = cross_val_score(model, x, y, cv=5)
    print("Cross validation is :",np.mean(score)*100)


# model = LogisticRegression()
# classify(model, x, y)

# model = DecisionTreeClassifier()
# classify(model, x, y)

# model = RandomForestClassifier()
# classify(model, x, y)

# model = ExtraTreesClassifier()
# classify(model, x, y)

model = LogisticRegression()
model.fit(x_trian, y_train)
y_pred = model.predict(x_test)

# Corelation Matric

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
# plt.show()




# new_data = pd.read_csv("test.csv")
# outcome = model.predict(new_data)
# print("Outcomes :", outcome)


