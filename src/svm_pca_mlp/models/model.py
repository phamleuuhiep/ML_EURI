import torch.nn as nn

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