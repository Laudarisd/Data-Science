import torch
from preprocessing_data import ImportData
from sklearn.model_selection import train_test_split

import warnings

__author__ = "SD"
__Date__ = "2023/08/01"

warnings.filterwarnings('ignore')

class SplitTrainTest(ImportData):
    def __init__(self, root_files):
        super(SplitTrainTest, self).__init__(root_files)
        self.root_files = root_files
        self.data = self.read_all_files()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()

    def split_train_test(self):
        X, y = self.data_cleaning(self.data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        # train-test split of the dataset
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)
        return X_train, X_test, y_train, y_test
    

if __name__ == "__main__":
    root_files = './data/'
    data_splitter = SplitTrainTest(root_files)
    X_train, X_test, y_train, y_test = data_splitter.X_train, data_splitter.X_test, data_splitter.y_train, data_splitter.y_test

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(y_train.unique())
    print(y_test.unique())
    print(y_train.isnull().sum())
    print(y_test.isnull().sum())
