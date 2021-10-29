# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierCNN(nn.Module):
    def __init__(self, tokenizer_type, vocab_size, max_len, class_num, device, embed_size=300, filter_num=128, filter_size=5, linear_size=20):
        super(ClassifierCNN, self).__init__()

        """
        Initialize Text Classifier CNN model
        Args:
            tokenizer_type (str): Type of tokenizer to use.
            vocab_size (int): Size of vocabulary.
            max_len (int): Maximum length of input sequence.
            class_num (int): Number of classes.
            device (torch.device): Device to run the model on
            filter_num (int): Number of filters in the CNN
            filter_size (int): Size of the filters in the CNN
            linear_size (int): Size of the linear layer
        """
        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.device = device
        self.embed_size = embed_size
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.linear_size = linear_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_size, out_channels=filter_num, kernel_size=filter_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.linear_input_size = filter_num * ((embed_size - filter_size + 1) // 2)
        self.linear1 = nn.Sequential(
            nn.Linear(self.linear_input_size, linear_size),
            nn.ReLU()
        )
        self.linear2 = nn.Linear(linear_size, class_num)
        nn.init.normal_(self.linear2.weight)
        nn.init.normal_(self.linear2.bias)

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
        # Reshape embedding
        embed = embed.permute(0, 2, 1) # (batch_size, embed_size, max_len)
        # Convolution
        conv = self.conv(embed)
        # Reshape conv
        conv = conv.view(conv.size(0), -1)
        # Linear
        linear = self.linear1(conv)
        output = self.linear2(linear)
        return output