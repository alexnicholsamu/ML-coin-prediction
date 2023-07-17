import torch.nn as nn


class DFFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DFFNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.PReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change dimensions to (batch_size, input_dim, sequence_length)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(x)  # Reduce sequence length to 1
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.act(self.bn2(self.fc1(x)))
        x = self.dropout(x)
        x = self.act(self.bn3(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

input_dim = 3
hidden_dim = 256
output_dim = 1
model = DFFNN(input_dim, hidden_dim, output_dim)


def getModel():
    return model
