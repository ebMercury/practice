import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# read file
df = pd.read_csv('/content/drive/MyDrive/loans.csv')

# get some information about data 

print(df.head())

print(df.columns)

print(df['loan_type'].unique())

print(df.info())

# outliers

Q1 = df['loan_amount'].quantile(0.25)
Q3 = df['loan_amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['loan_amount'] < (Q1 - 1.5 * IQR)) | (df['loan_amount'] > (Q3 + 1.5 * IQR)))]

# change data type of datetime columns

df['loan_start'] = pd.to_datetime(df['loan_start'], errors='coerce')

df['loan_end'] = pd.to_datetime(df['loan_end'], errors='coerce')

# check for duplicate values or nan values

print(df.duplicated().sum())

print(df.isna().sum())

#feature engineering

df['loan_duration'] = (df['loan_end'] - df['loan_start']).dt.days
df.drop(['loan_start', 'loan_end'], axis=1, inplace=True)

# delete columns

df.drop(['client_id', 'loan_id'], axis=1, inplace=True)

# one_hot encoding
df = pd.get_dummies(df, columns=['loan_type'], drop_first=True, dtype='int')


# scaling : Standard

standard = StandardScaler()
columns = ['loan_amount', 'loan_duration']
for column in columns:
  df[column] = standard.fit_transform(df[[column]])


# split data into train and test
# rate is the target

X = df.drop('rate', axis=1)
y = df['rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)