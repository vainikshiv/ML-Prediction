import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import os

# Read CSV file
dataset = pd.read_csv('dummydata.csv')

# Add new column
col = input('Do you want to add new predictor? (Y/N):')
if col in ['Y','y']:
    name = input("Enter new predictor name : ")
    l = list(input("Provide a column data (e.g. 3500,3543,3521 etc) :").split(','))
    additional = pd.DataFrame({ name: l })
    new = pd.concat([dataset, additional], axis=1)
    new[name].fillna(0, inplace=True)
    os.remove('dummydata.csv')
    new.to_csv('dummydata.csv', index=False)

#Add new row into dataset
new_row = {}
dataset = pd.read_csv('dummydata.csv')
data_list = [k for k in dataset.columns.values if k != 'Actual']

print('\nEnter below values for prediction.')
for i in data_list:
    enter = input(f'Enter {i} :')
    if enter != '':
        if i == 'date':
            new_row[i] = enter
        else:
            new_row[i] = float(enter)
    else:
        new_row[i] = 0.0

# Actual value prediction function
def get_actual_value(data,row):
    X = data[[j for j in list(data.columns.values) if j not in ['date', 'Actual']]].values
    y = data['Actual'].values

    # Train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)

    # print(regressor.coef_)

    act = regressor.predict([[row[i] for i in row if i != 'date']])
    print('-'*80,'\n Actual value: ', act[0])
    row['Actual'] = act[0]
    data = data.append(row, ignore_index=True)

    # Create new csv file and append new data to it
    os.remove('dummydata.csv')
    data.to_csv('dummydata.csv', index=False)

# Call get_actual_value function
get_actual_value(dataset, new_row)
