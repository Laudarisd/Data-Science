import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')



#make class for model

class ModelKNN:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    def model(self):
        #train model
        #knn = KNeighborsClassifier(n_neighbors=5)
        #grid search
        param_grid = {  'n_neighbors': [5, 10, 15, 20, 25, 30],
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                        'leaf_size': [10, 20, 30, 40, 50, 60],
                        'p': [1, 2],
                        'metric': ['minkowski', 'euclidean', 'manhattan'],
                        'metric_params': [None],
                        'n_jobs': [-1]}

        knn = KNeighborsClassifier()
        self.grid_search = GridSearchCV(knn, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
        self.grid_search.fit(self.X_train, self.y_train)
        #predict
        self.y_pred = self.grid_search.predict(self.X_test)
        #print best parameters
        self.best_param = []
        for key, value in self.grid_search.best_params_.items():
            self.best_param.append(key + ': ' + str(value))
            print(key, value)
        self.acc_each_fold = []
        for i in range(5):
            self.acc_each_fold.append(self.grid_search.cv_results_['split' + str(i) + '_test_score'][self.grid_search.best_index_])
        #evaluate model
        print('Accuracy: ', accuracy_score(self.y_test, self.y_pred))
        print('F1 score: ', f1_score(self.y_test, self.y_pred, average='weighted'))
        print('Confusion matrix: ', confusion_matrix(self.y_test, self.y_pred))
        print('Classification report: ', classification_report(self.y_test, self.y_pred))
    def actaul_vs_prediction_dataframe(self):
        #create dataframe to compare actual vs predicted
        self.df = pd.DataFrame({'Actual': self.y_test, 'Predicted': self.y_pred})
        print(self.df)
    def save_allresults(self):
        save_path = './result/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + 'knn.log', 'a') as f:
            f.write(" Knn Classifier")
            f.write('\n')
            f.write("################ Best Parameters ################")
            f.write('\n')
            for i in self.best_param:
                f.write(i)
                f.write('\n')
            f.write("################ Accuracy ################")
            f.write('\n')
            f.write('Accuracy: ' + str(accuracy_score(self.y_test, self.y_pred)))
            f.write('\n')
            f.write("################ Accuracy from each fold ################")
            f.write('\n')
            for i in self.acc_each_fold:
                f.write('Accuracy: ' + str(i))
                f.write('\n')
            f.write("################ F1 score ################")
            f.write('\n')
            f.write('F1 score: ' + str(f1_score(self.y_test, self.y_pred, average='weighted')))
            f.write('\n')
            f.write("################ Confusion matrix ################")
            f.write('\n')
            f.write('Confusion matrix: ' + str(confusion_matrix(self.y_test, self.y_pred)))
            f.write('\n')
            f.write("################ Classification report ################")
            f.write('\n')
            f.write('Classification report: ' + str(classification_report(self.y_test, self.y_pred)))
            f.write('\n')
            f.write("################ Actual vs Predicted ################")
            f.write('\n')
            f.write(str(self.df))
            f.write('\n')
          