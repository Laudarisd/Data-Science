import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



__author__ ="SD"
__Date__ = "2023/08/01"

class ImportData:
    def __init__(self, root_files):
        self.root_files = root_files
        self.data = self.read_all_files()

    def read_all_files(self):
        all_data = pd.DataFrame()
        for file_path in os.listdir(self.root_files):
            full_file_path = os.path.join(self.root_files, file_path)
            if os.path.isfile(full_file_path) and file_path.endswith('.csv'):
                data = pd.read_csv(full_file_path)
                all_data = pd.concat([all_data, data], ignore_index=True)
        return all_data

    def data_cleaning(self, data):
        select_radius = [0.011, 0.015, 0.020, 0.025, 0.029]
        data = data.loc[data['radius'].isin(select_radius)]
        X = data.drop(columns=['radius'])
        X = X.reindex(sorted(X.columns), axis=1)
        X = X.iloc[:, 10:]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = data['radius']
        y = y.map({0.011: 0, 0.015: 1, 0.020: 2, 0.025: 3, 0.029: 4})

        return X, y

if __name__ == "__main__":
    root_files = './data/'
    data_loader = ImportData(root_files)
    data = data_loader.data
    X, y = data_loader.data_cleaning(data)
