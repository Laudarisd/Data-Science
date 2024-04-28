import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



class ModelMLP():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    def model(self):
        mlp = MLPClassifier(hidden_layer_sizes=(10, ), max_iter=100, alpha=0.001,
                     solver='adam', verbose=10,  random_state=21,tol=0.0001)
        mlp.fit(self.X_train, self.y_train)
        self.y_pred = mlp.predict(self.X_test)
        print("Multi-layer Perceptron")
        print('Accuracy: ', accuracy_score(self.y_test, self.y_pred))
        print('F1 score: ', f1_score(self.y_test, self.y_pred, average='weighted'))
        print('Confusion matrix: ', confusion_matrix(self.y_test, self.y_pred))
        print('Classification report: ', classification_report(self.y_test, self.y_pred))
        #learning curve
        train_sizes, train_scores, test_scores = learning_curve(mlp, self.X_train, self.y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.grid()
        plt.xlabel('Number of training samples')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()
    def actaul_vs_prediction_dataframe(self):
        self.df = pd.DataFrame({'Actual': self.y_test, 'Predicted': self.y_pred})
        print(self.df)