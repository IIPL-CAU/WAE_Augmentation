from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

class ClassifierBERT(BertForSequenceClassification):
    def __init__(self, isPreTrain):
        super(ClassifierBERT, self).__init__()
        if isPreTrain:
            self.model = BertForSequenceClassification.from_pretrained('bert-base-cased')
        else:
            model_config = BertConfig('bert-base-cased')
            self.model = BertForSequenceClassification(config=model_config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        out = out['logits']

        return out