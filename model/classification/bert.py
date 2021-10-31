import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

class ClassifierBERT(nn.Module):
    def __init__(self, isPreTrain, num_class):
        super(ClassifierBERT, self).__init__()
        if isPreTrain:
            self.model = BertForSequenceClassification.from_pretrained('bert-base-cased')
            self.model.classifier = nn.Linear(768, num_class)
        else:
            model_config = BertConfig('bert-base-uncased')
            self.model = BertForSequenceClassification(config=model_config)
            self.model.classifier = nn.Linear(768, num_class)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        out = out['logits']

        return out