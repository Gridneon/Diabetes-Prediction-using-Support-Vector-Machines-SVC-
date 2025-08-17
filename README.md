import numpy as np

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.preprocessing import  StandardScaler

from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix

from sklearn.impute import SimpleImputer

from sklearn import svm    # model that we are using 

#importing Diabetes the data from source 
import os
os.getcwd()
os.listdir(os.getcwd())
os.listdir(os.chdir('c:\\Users\\HP\\Downloads\\datascience.py\\data_files'))

diabetes_link = 'c:\\Users\\HP\\Downloads\\datascience.py\\data_files\\diabetes.csv'

diabetes_data =  pd.read_csv(diabetes_link)
diabetes_data.head(6)
#Data exploration 
#column, dim, structure and summary of data 

diabetes_data.size    # get the size of the data 
diabetes_data.columns   # column names in data 

diabetes_data.info()     # get the structure of the data 

diabetes_data.describe()    # describes the data 



#data Preparation 
#rename the columns
data=diabetes_data.rename(columns={'Pregnancies':'n_pregnancies', 
                             'Glucose':'glucose','BloodPressure':'bp', 
                             'SkinThickness':'skinthic','Insulin':'insulin',
                             'BMI':'bmi', 'DiabetesPedigreeFunction': 'dpf', 
                             'Age':'age', 'Outcome':'diab_present'})
data.head()


#duplicate
data.duplicated().sum()    #or 
data.duplicated().value_counts()

#is NAN,na
data.isna().value_counts()

#need to standaized the data == yes the insulin column
#get the insulin column
insulin_column = data.loc[:,"insulin"]
insulin_column.describe() 



data.head()

#check if there are missing values...in
#this case 0 are in each column 

#'Glucose','BloodPressure','SkinThickness','Insulin','BMI'
#cant have a 0 values this signifies invalid info,
#we need to replace this values with 
#check each value with 0's 

data[data['glucose'] == 0] #to identify the rows 

(data['glucose'] == 0).value_counts() #to get the number of zeros 

data[data['bp'] == 0] #to identify the rows 

(data['bp'] == 0).value_counts()#to get the number of zeros 

data[data['skinthic'] == 0] #to identify the rows 

(data['skinthic'] == 0).value_counts()#to get the number of zeros 

data[data['bmi'] == 0] #to identify the rows 

(data['bmi'] == 0).value_counts()#to get the number of zeros 


#dutribution of each selected column 
data.skew()

sns.histplot(data['bmi'])
sns.histplot(data["bp"])
sns.histplot(data["skinthic"])
sns.histplot(data["glucose"])
sns.histplot(data['skinthic'])



#imputing median for the 0 in the column above 
zero_values =['glucose','bp','skinthic','bmi']
imputer = SimpleImputer(missing_values=0,strategy="median")
data[zero_values] = imputer.fit_transform(data[zero_values])


data.describe()
(data[zero_values] == 0).sum()


#comparing the outcomes
data.head()
sns.boxplot(x= 'diab_present', y= "dpf",data=data) 


#scaling all continuous data to prepare data for machine learning model
                        # SVM
scale_data = ['n_pregnancies', 'glucose', 'bp', 'skinthic', 'bmi', 'dpf', 'age','insulin']

scaler = StandardScaler()

data[scale_data] =scaler.fit_transform(data[scale_data])


#split data into taining data set and Test dataset
 
X = data.drop("diab_present",axis = 1)
Y = data['diab_present']

data.shape
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,
                                                 random_state = 2)
print(X.shape,X_test.shape,X_train.shape)
print(Y.shape,Y_test.shape,Y_train.shape)

classifier = svm.SVC(kernel='linear')

#training the support vector classifer
classifier.fit(X_train,Y_train)

#testing the model 
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy :', training_data_accuracy)


#test prediction 
X_test_prediction = classifier.predict(X_test) 
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy :', test_data_accuracy,'the model has not overtrain hence there is no overfitting')


#building a prodictive system
input_data = (0,137, 40,35, 168,43,2,33)


#changing the input_data to numpy array
input_data_as_numpy_array = np.array(input_data)
#reshape
reshape_input_data_as_numpy_array = input_data_as_numpy_array.reshape(1,-1)
#standardize datapoint 


std_data1 = scaler.transform(reshape_input_data_as_numpy_array )

#####predicting a data point 
prediction = classifier.predict(std_data1)
print(prediction)

diabetes_data.iloc[4,:]
