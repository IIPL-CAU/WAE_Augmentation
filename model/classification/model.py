from model.classification.cnn import ClassifierCNN
from model.classification.rnn import ClassifierRNN
from model.classification.bert import ClassifierBERT

class Classifier(nn.Module):
    def __init__(self, model_type, isPreTrain):
        super().__init__()

        self.model_type = model_type

        if model_type == 'CNN':
            self.model = ClassifierCNN()

        if model_type == 'RNN':
            self.model = ClassifierRNN()

        if model_type == 'BERT':
            self.model = ClassifierBERT()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)

        return out