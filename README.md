# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SANJEV R M
RegisterNumber: 212223040186
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
print(mse)
r2=metrics.r2_score(y_test,y_pred)
print(r2)
dt.predict([[5,6]])
```
## Output:
# Data.Head():
![image](https://github.com/user-attachments/assets/e3c604a7-74d9-4f98-a974-7a019c2b7dee)
# Data.info():
![image](https://github.com/user-attachments/assets/d119e1f9-4933-40b9-a43c-769012043d71)
# isnull() and sum():
![image](https://github.com/user-attachments/assets/602b324a-8235-4852-8ff5-3b6f8d8567cf)
# MSE Value:
![image](https://github.com/user-attachments/assets/5b97b008-81f0-4c2a-aaa0-0e96f238112d)
# R2 Value:
![image](https://github.com/user-attachments/assets/07ca2c5b-6288-4048-ad1e-d0041885877e)
# Data Prediction:
![image](https://github.com/user-attachments/assets/b6526b51-872c-4a3a-a516-f44a0d388a4a)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
