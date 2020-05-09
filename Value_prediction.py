import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from csv import reader, writer
 
file_path = 'dummydata.csv'

# Read CSV file
dataset = pd.read_csv('dummydata.csv')

#Add new row into dataset
new_row = {}

print('Enter below values for prediction.')
for i in ['date','Alpha','Bravo','Charlie','Delta','Echo','Foxtrot','Golf', 'Hotel', 'India','Juliett']:
    enter = input(f'Enter {i} :')
    if enter is not None:
        if i == 'date':
            new_row[i] = enter
        else: 
            new_row[i] = float(enter)


X = dataset[['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot', 'Golf', 'Hotel', 'India','Juliett']].values
y = dataset['Actual'].values

# Train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

# print(regressor.coef_)

act = regressor.predict([[new_row['Alpha'], new_row['Bravo'], new_row['Charlie'], new_row['Delta'],new_row['Echo'], new_row['Foxtrot'], new_row['Golf'], new_row['Hotel'], new_row['India'],new_row['Juliett']]])
print('-'*80,'\n Actual value: ', act[0])
new_row['Actual'] = act[0]
dataset = dataset.append(new_row, ignore_index=True)

# Create new csv file and append new data to it
dataset.to_csv('raw_output.csv', index=False)
