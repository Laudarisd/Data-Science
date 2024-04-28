import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')


#make class for model

class DecissionTree():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    def model(self):
        #train model
        self.dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
        self.dt.fit(self.X_train, self.y_train)
        #predict
        self.y_pred = self.dt.predict(self.X_test)
        #evaluate model
        print("Decision Tree")
        print('Accuracy: ', accuracy_score(self.y_test, self.y_pred))
        print('F1 score: ', f1_score(self.y_test, self.y_pred, average='weighted'))
        print('Confusion matrix: ', confusion_matrix(self.y_test, self.y_pred))
        print('Classification report: ', classification_report(self.y_test, self.y_pred))
    def actaul_vs_prediction_dataframe(self):
        #create dataframe to compare actual vs predicted
        self.df = pd.DataFrame({'Actual': self.y_test, 'Predicted': self.y_pred})
        print(self.df)
    def feature_importance(self):
        #get feature importance
        self.feature_importances = pd.DataFrame(self.dt.feature_importances_,
                                   index = self.X_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
        print(self.feature_importances)