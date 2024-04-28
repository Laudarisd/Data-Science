import torch
import torch.nn as nn
from torchsummary import summary
from torchviz import make_dot



__author__ ="SD"
__Date__ = "2023/08/01"

class ModelTorch(nn.Module):
    def __init__(self):
        super(ModelTorch, self).__init__()

        # Define the model architecture
        self.model = nn.Sequential(
            nn.Linear(70, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )

    def forward(self, x):
        # Define the forward pass
        return self.model(x)

    def visualize_model(self):
        # Visualize the model graph
        x = torch.randn(1, 70)
        out = self.model(x)
        make_dot(out, params=dict(self.model.named_parameters())).view()
    def model_summary(self):
        # Get a summary of all layers in your network
        summary(self.model, (1, 70))

    
if __name__ == "__main__":
    n_epochs = 10
    batch_size = 64
    model_obj = ModelTorch(n_epochs, batch_size)
    # Visualize the model graph
    model_obj.visualize_model()
    # Get a summary of all layers in your network
    model_obj.model_summary()

