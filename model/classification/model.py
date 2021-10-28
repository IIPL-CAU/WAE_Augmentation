from cnn import CNN_model
from rnn import RNN_model
from bert import BERT_model

class cls_model(nn.Module):
    def __init__(self, model_type, isPreTrain):
        super().__init__()

        self.model_type = model_type

        if model_type == 'BERT':
            if isPreTrain:
                self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
            else:
                model_config = BertConfig('bert-base-uncased')
                self.model = BertForSequenceClassification(config=model_config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        if self.model_type == 'BERT':
            out = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
            out = out['logits']
        
        if self.model_type == 'CNN':
            out = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        
        if self.model_type == 'RNN':
            out = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        return out