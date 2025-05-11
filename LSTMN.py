import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers):
        super(LSTMNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        x = self.embedding(x)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1]
        out = self.fc(lstm_out)
        return self.sigmoid(out), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if torch.cuda.is_available():
            hidden = (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_().cuda(),
                      weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_(),
                      weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_())
        return hidden

