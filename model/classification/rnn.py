# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ClassifierRNN(nn.Module):
    def __init__(self, vocab_size, max_len, class_num, embed_size=300, hidden1_size=64, hidden2_size=32, linear_size=20):
        super(ClassifierRNN, self).__init__()

        """
        Initialize Text Classifier CNN model
        Args:
            vocab_size (int): Size of vocabulary.
            max_len (int): Maximum length of input sequence.
            class_num (int): Number of classes.
            filter_num (int): Number of filters in the CNN
            filter_size (int): Size of the filters in the CNN
            linear_size (int): Size of the linear layer
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_size = embed_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.linear_size = linear_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm1 = nn.LSTM(embed_size, hidden1_size, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(2 * hidden1_size, hidden2_size, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.linear_input_size = 2 * hidden2_size
        self.linear1 = nn.Sequential(
            nn.Linear(self.linear_input_size, linear_size),
            nn.ReLU()
        )
        self.linear2 = nn.Linear(linear_size, class_num)
        nn.init.normal_(self.linear2.weight)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass of the model
        Args:
            input_ids (torch.tensor): Input tensor of shape (batch_size, max_len)
        Returns:
            torch.tensor: Output tensor of shape (batch_size, class_num)
        """
        # Embedding
        embed = self.embedding(input_ids) # (batch_size, max_len, embed_size)
        # Pack padded sequence
        non_pad_len = input_ids.ne(self.tokenizer.pad_token_id).sum(dim=1).cpu()
        # LSTM 1
        packed_embed = pack_padded_sequence(embed, non_pad_len, batch_first=True, enforce_sorted=False)
        lstm1_out, _ = self.lstm1(packed_embed) # (batch_size, max_len, 2 * hidden1_size)
        lstm1_out, _ = pad_packed_sequence(lstm1_out, batch_first=True) # (batch_size, max_len, 2 * hidden1_size)
        lstm1_out = self.dropout1(lstm1_out)
        # LSTM 2
        packed_lstm1_out = pack_padded_sequence(lstm1_out, non_pad_len, batch_first=True, enforce_sorted=False)
        lstm2_out, _ = self.lstm2(packed_lstm1_out) # (batch_size, max_len, 2 * hidden2_size)
        lstm2_out, _ = pad_packed_sequence(lstm2_out, batch_first=True) # (batch_size, max_len, 2 * hidden2_size)
        lstm2_out = self.dropout2(lstm2_out)
        # Linear
        linear = self.linear1(lstm2_out[:, -1, :]) # (batch_size, linear_size)
        # Softmax
        output = self.linear2(linear) # (batch_size, class_num)
        return output