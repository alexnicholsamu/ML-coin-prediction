import torch.nn as nn
import torch


class DFFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DFFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


input_dim = 1
hidden_dim = 256
output_dim = 1
model = DFFNN(input_dim, hidden_dim, output_dim)

def getModel():
    return model
