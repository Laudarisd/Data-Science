'''
Data preprocessing code
'''
__aurthor__ = "Laudari Sudip"
__python_version__ = "3.10.6"

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")






# class Preprocessing(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass

#     def __init__(self, X, y):
#         self.X = X
#         self.y = y
#     def fill_missing_values(self):
#         #fill missing values with 0
#         self.X = self.X.fillna(0)
#         return self.X
#     def Convert_y_to_numeric(self):
#         #convert y to numeric
#         labelencoder = LabelEncoder()
#         self.y = labelencoder.fit_transform(self.y)
#         return self.y
#     def split(self):
#         #split data into train and test
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1)
#         print("X_tran shape: ", self.X_train.shape)
#         print("X_test shape: ", self.X_test.shape)
#         print("y_train shape: ", self.y_train.shape)
#         print("y_test shape: ", self.y_test.shape)
#         print("Training features: ", self.X_train.columns)
#         return self.X_train, self.X_test, self.y_train, self.y_test
#     def normalize(self):
#         #normalize data
#         self.X_train = (self.X_train - self.X_train.min()) / (self.X_train.max() - self.X_train.min())
#         self.X_test = (self.X_test - self.X_test.min()) / (self.X_test.max() - self.X_test.min())
#         return self.X_train, self.X_test
#     # def standardize(self):
#     #     #standardize data
#     #     self.X_train = (self.X_train - self.X_train.mean()) / self.X_train.std()
#     #     self.X_test = (self.X_test - self.X_test.mean()) / self.X_test.std()
#     #     return self.X_train, self.X_test    
    