import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model import ModelTorch
from split_data import SplitTrainTest
from sklearn.metrics import mean_squared_error, r2_score


class TrainData:
    def __init__(self, root_files, n_epochs, batch_size, learning_rate):
        self.root_files = root_files
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def train_model(self):
        # Instantiate the data splitter class
        data_splitter = SplitTrainTest(self.root_files)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = data_splitter.X_train, data_splitter.X_test, data_splitter.y_train, data_splitter.y_test

        # Instantiate the model class
        model = ModelTorch()

        # loss function and optimizer
        loss_fn = nn.MSELoss()  # mean square error
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # training parameters
        batch_start = torch.arange(0, len(X_train), self.batch_size)

        # Lists to store train and test losses for plotting
        train_losses = []
        test_losses = []

        # Hold the best model and best MSE
        best_mse = np.inf
        best_weights = None

        # training loop
        for epoch in range(self.n_epochs):
            model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = X_train[start:start+self.batch_size]
                    y_batch = y_train[start:start+self.batch_size]
                    # forward pass
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    bar.set_postfix(mse=float(loss))

            # evaluate accuracy at end of each epoch
            # Evaluate accuracy at the end of each epoch
            model.eval()
            with torch.no_grad():
                y_pred_train = model(X_train)
                y_pred_test = model(X_test)
                train_loss = loss_fn(y_pred_train, y_train)
                test_loss = loss_fn(y_pred_test, y_test)

            # Append train and test loss to their respective lists
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())

            # Check if the current model has the best MSE so far
            if test_loss < best_mse:
                best_mse = test_loss
                best_weights = copy.deepcopy(model.state_dict())

        # Load the best model weights
        model.load_state_dict(best_weights)

        print("Best MSE: %.2f" % best_mse)
        print("Best RMSE: %.2f" % np.sqrt(best_mse))

        # Plot train and test losses
        plt.plot(range(1, self.n_epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, self.n_epochs + 1), test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./result/learning_curve.png', dpi=300)
        plt.show()
        
        # Create a DataFrame of true y_test and predictions
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test)
        y_pred_test = y_pred_test.detach().numpy()
        y_test = y_test.detach().numpy()
        df = pd.DataFrame({'True y_test': y_test.flatten(), 'Predicted y_test': y_pred_test.flatten()})
        df.to_csv('./result/predictions.csv', index=False)
        predictions_df = pd.read_csv('./result/predictions.csv')

        # Calculate MSE, RMSE, and R2 for training and test sets
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        # Create a DataFrame with MSE, RMSE, and R2 values
        mse_rmse_r2_table = pd.DataFrame({
            'Training': [train_mse, train_rmse, train_r2],
            'Test': [test_mse, test_rmse, test_r2]
        }, index=['MSE', 'RMSE', 'R2'])

        # Print the table
        print("MSE, RMSE, and R2 for Training and Test Sets:")
        print(mse_rmse_r2_table)
        unique_classes = predictions_df['True y_test'].value_counts()
        print("Unique prediction class counts:")
        print(unique_classes)

        # Plot the predictions
        plt.scatter(predictions_df['True y_test'], predictions_df['Predicted y_test'])
        plt.xlabel('True y_test')
        plt.ylabel('Predicted y_test')
        plt.savefig('./result/predictions.png', dpi=300)
        plt.show()
        return
    

if __name__ == "__main__":
    root_files = './data/'
    train_data = TrainData(root_files, 500, 16, 0.00001)
    train_data.train_model()
