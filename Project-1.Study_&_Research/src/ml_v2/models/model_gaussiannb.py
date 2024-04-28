import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')



class ModelGaussianNB():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    def model(self):
        gnb = GaussianNB()
        gnb.fit(self.X_train, self.y_train)
        self.y_pred = gnb.predict(self.X_test)
        print("Gaussian Naive Bayes")
        print('Accuracy: ', accuracy_score(self.y_test, self.y_pred))
        print('F1 score: ', f1_score(self.y_test, self.y_pred, average='weighted'))
        print('Confusion matrix: ', confusion_matrix(self.y_test, self.y_pred))
        print('Classification report: ', classification_report(self.y_test, self.y_pred))
    def actaul_vs_prediction_dataframe(self):
        self.df = pd.DataFrame({'Actual': self.y_test, 'Predicted': self.y_pred})
        print(self.df)
    
   