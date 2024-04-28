'''
Data preprocessing code
'''
__aurthor__ = "Laudari Sudip"
__python_version__ = "3.10.6"

from multiprocessing.sharedctypes import Value
from pyexpat import model
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


#import models
from models.model_rf import ModelRF



import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('./data/original_csv.csv')

SEED = 42
TEST_SIZE = 0.2

#Seems there are '?' in Bare Nuclei column. Fill it with Nan

X = df.drop(['class'], axis=1) #, 'Bare Nuclei'
# print only missinf, unrecognized entries
print(X[X == '?'].count())
#replace ? with most frequent value
# for col in X.columns:
#     X[col] = X[col].replace('?', X[col].value_counts().index[0])
# print(X[X == '?'].count())

le = LabelEncoder()
 
# Using .fit_transform function to fit label
# encoder and return encoded label
label = le.fit_transform(X['Bare Nuclei'])
#X = X.drop(['Bare Nuclei'], axis=1, inplace=True)
X['Bare Nuclei'] = label
# printing label
#X = (X - X.mean()) / X.std()
#print(X)
#standardize data
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
#replace NaN with 0
# X = X.fillna(0)
print(X[X== '?'].count())
#print(X)
y = df['class']


class DataPipeline:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def labelencoder(self):
        labelencoder = LabelEncoder()
        self.y = labelencoder.fit_transform(self.y)
        #print(self.y)
        return self.y
    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=TEST_SIZE, random_state=SEED)
        #print(self.X_train.head(50))
        #self.X_train = self.X_train.fillna(0)
        print(self.X_train.shape)
        return self.X_train, self.X_test, self.y_train, self.y_test
    #createa pipeline for numerical and categorical values
    def pipeline(self):
        self.numerical_features = self.X_train.select_dtypes(include='number').columns.tolist()
        print(f'There are {len(self.numerical_features)} numerical features:', '\n')
        print(self.numerical_features)
        self.categorical_features = self.X_train.select_dtypes(exclude='number').columns.tolist()
        print(f'There are {len(self.categorical_features)} categorical features:', '\n')
        print(self.categorical_features)
        #self.categorical_features = OneHotEncoder(handle_unknown='ignore')
        #Following pipeline will input missing values, and scale X_train
        self.numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scale', MinMaxScaler())
        ])
        self.categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            #('imputer', SimpleImputer(missing_values='?', strategy='mean')),
            #('imputer', SimpleImputer(missing_values= 'nan', strategy= 'median')),
            ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        try:
            self.full_processor  = ColumnTransformer(transformers=[
                                ('number', self.numeric_pipeline, self.numerical_features),
                                ('category', self.categorical_pipeline, self.categorical_features)
                            ])
            print(self.full_processor.fit_transform(self.X_train))
        except ValueError:
            print("Error occured: Check Pipeline")
    def lasso_estimator(self):
        self.lasso = Lasso(alpha=0.1)

        self.lasso_pipeline = Pipeline(steps=[
            ('preprocess', self.full_processor),
            ('model', self.lasso)
        ])
        try:
            self.model_fit = self.lasso_pipeline.fit(self.X_train, self.y_train)
            self.y_pred = self.model_fit.predict(self.X_test)
            self.mae = round(mean_absolute_error(self.y_test, self.y_pred), 3)
            print(f'Lasso Regression - MAE: {self.mae}')
            return self.lasso_pipeline
        except ValueError:
            print("Error occured while training lasso model")

    def rf_estiimator(self):
        self.rf_model =  RandomForestClassifier()
        self.rf_pipeline = Pipeline(steps=[
            ('preprocess', self.full_processor),
            ('model', self.rf_model)
        ])
        try:
            self.rf_model_fit = self.rf_pipeline.fit(self.X_train, self.y_train)
            self.y_pred = self.rf_model_fit.predict(self.X_test)
            #get feature importance
            print(self.rf_pipeline[:-1].get_feature_names_out())
            print(self.rf_model_fit[-1].feature_importances_)

            self.features_importance = pd.DataFrame({'features': self.rf_pipeline[:-1].get_feature_names_out(), 'importance': self.rf_model_fit[-1].feature_importances_})
            self.features_importance = self.features_importance.sort_values(by='importance', ascending=False)
            print(self.features_importance)
            print("Accuracy:", accuracy_score(self.y_test, self.y_pred))
            print("F1 score:", round(f1_score(self.y_test, self.y_pred), 3))
        except ValueError:
            print("Error occured while training random forest model")
    def run(self):
        self.labelencoder()
        self.split()
        self.pipeline()
        self.lasso_estimator()
        self.rf_estiimator()

T = DataPipeline(X, y)
print(T.run())



    