import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')


#make class for model

class ModelRF:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    def model(self):
        #grid serach
        param_grid = {  'n_estimators': [100, 200, 300, 400, 500],
                        'max_depth': [5, 10, 15, 20, 25, 30],
                        'random_state': [0, 1, 42],
                        'criterion': ['gini', 'entropy'],
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'bootstrap': [True, False],
                        'oob_score': [True, False],
                        'class_weight': ['balanced', 'balanced_subsample', None],
                        'warm_start': [True, False],
                        #'n_jobs': [-1],
                        'verbose': [0, 1, 2],
                        'ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                       
                        }
        self.rf = RandomForestClassifier()
        self.random_search = RandomizedSearchCV(self.rf, 
                                                param_grid, 
                                                n_iter=100, 
                                                cv=5, 
                                                verbose=2, 
                                                random_state=1, 
                                                n_jobs=-1)
        print("Random Forest")
        #print each best parameter in new line
        self.best_parameters = []
        for key, value in self.rf.get_params().items():
            self.best_parameters.append(key + ': ' + str(value))
            print(key, value)
        best_param = self.random_search.fit(self.X_train, self.y_train)
        print("################ Best Parameters ################")
        for key, value in best_param.best_params_.items():
            self.best_parameters.append(key + ': ' + str(value))
            print(key, value)
        self.y_pred = self.random_search.predict(self.X_test)
        print("################ Accuracy ################")
        print('Accuracy: ' + str(accuracy_score(self.y_test, self.y_pred)))
        #print accuracy from each fold
        self.acc_each_fold = []
        for i in range(5):
            self.acc_each_fold.append(self.random_search.cv_results_['split' + str(i) + '_test_score'][self.random_search.best_index_])
            print('Accuracy: ' + str(self.random_search.cv_results_['split' + str(i) + '_test_score'][self.random_search.best_index_]))
        #evaluate model
        print("Random Forest")
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
        self.feature_importances = pd.DataFrame(self.random_search.best_estimator_.feature_importances_,
                                      index = self.X_train.columns,
                                        columns=['importance']).sort_values('importance', ascending=False)
        print(self.feature_importances)

    def save_allresults(self):
        save_dir = './result/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        #save all results in log file
        with open(save_dir + 'rf.log', 'a') as f:
            f.write("Random Forest")
            f.write('\n')
            f.write("################ Best Parameters ################")
            f.write('\n')
            for i in self.best_parameters:
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
            f.write("################ Feature Importance ################")
            f.write('\n')
    def plot_roc_curve_with_kfold(self, fprs, tprs):
        """Plot the Receiver Operating Characteristic from a list
        of true positive rates and false positive rates."""
        
        # Initialize useful lists + the plot axes.
        tprs_interp = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        f, ax = plt.subplots(figsize=(14,10))
        #plot ROC for each k-fold + compute AUC acores
        for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
            tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
            tprs_interp[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, lw=2, alpha=0.3,
                    label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        #plot the luck line
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
        #plot the mean ROC curve
        mean_tpr = np.mean(tprs_interp, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)
        #plot the standard deviation 
        std_tpr = np.std(tprs_interp, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
        #plot the axis labels
        # Fine tune and show the plot.
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic')
        ax.legend(loc="lower right")
        plt.show()
        return (f, ax)

    def compute_roc_auc(self, index):
        """Compute the ROC AUC score for the model."""
        y_prediction = self.random_search.predict_proba(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, y_prediction[:,1])
        roc_auc = auc(fpr, tpr)
        return (fpr, tpr, roc_auc)
    def plot_roc_curve(self):
        """Plot the Receiver Operating Characteristic curve."""
        fpr, tpr, roc_auc = self.compute_roc_auc(1)
        f, ax = plt.subplots(figsize=(14,10))
        ax.plot(fpr, tpr, lw=2, alpha=0.3,
                label='ROC (AUC = %0.2f)' % (roc_auc))
        #plot the luck line
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
        #plot the axis labels
        # Fine tune and show the plot.
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic')
        ax.legend(loc="lower right")
        plt.savefig('ROC.png')
        #plt.show()
        return (f, ax)