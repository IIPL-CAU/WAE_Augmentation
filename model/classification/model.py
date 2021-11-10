# Import PyTorch
import torch.nn as nn
# Import Custom Modules
from model.classification.cnn import ClassifierCNN
from model.classification.rnn import ClassifierRNN
from model.classification.bert import ClassifierBERT

class Classifier(nn.Module):
    def __init__(self, model_type, isPreTrain, num_class, tokenizer_type=None):
        super().__init__()

        self.model_type = model_type

        if model_type == 'CNN':
            self.model = ClassifierCNN(tokenizer_type=tokenizer_type,
                                       num_class=num_class)

        if model_type == 'RNN':
            self.model = ClassifierRNN(tokenizer_type=tokenizer_type,
                                       num_class=num_class)

        if model_type == 'BERT':
            self.model = ClassifierBERT(isPreTrain=isPreTrain, num_class=num_class)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)

        return out