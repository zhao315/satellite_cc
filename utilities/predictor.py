import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

def predictor():
    # load model
    model = torch.load("saved_model/predictor/linear_model.pt", weights_only=False)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import torch

    test_input = torch.randn(1, 384).to(device)
    model = predictor()
    model.to(device)
    
    output = model(test_input)
    print(f"output: {output}")