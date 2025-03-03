import torch
import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self._make_block(3, 8, 3)
        self.conv2 = self._make_block(8, 16, 3)
        self.conv3 = self._make_block(16, 32, 3)
        self.conv4 = self._make_block(32, 64, 3)
        self.conv5 = self._make_block(64, 64, 3)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=3136, out_features=512),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def _make_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class MyRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=2, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
        self.rnn1 = self._make_rnn_block(embedding_dim, hidden_size, num_layers)
        self.rnn2 = self._make_rnn_block(hidden_size, hidden_size, num_layers)
        self.rnn3 = self._make_rnn_block(hidden_size, hidden_size, num_layers)
        
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 128),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.fc3 = nn.Linear(64, num_classes)
        
    def _make_rnn_block(self, input_size, hidden_size, num_layers):
        return nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3)
    
    def forward(self, x):
        x = self.embedding(x)  
        x, _ = self.rnn1(x)  
        x, _ = self.rnn2(x)  
        x, _ = self.rnn3(x)  
        x = x[:, -1, :]    
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x