import torch
import torch.nn as nn
from utils.speech_to_text.preprocess import CHAR_VOCAB

class SpeechRNNCTC(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=512, output_dim=len(CHAR_VOCAB)):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=3,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.fc(x)
