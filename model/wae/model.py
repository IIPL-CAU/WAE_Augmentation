import math
# Import PyTorch
import torch
import torch.nn as nn
# Import Huggingface
# T5
from transformers import T5ForConditionalGeneration, T5EncoderModel, T5Config, T5Tokenizer
# Bart
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
# BERT
from transformers import BertTokenizer

class TransformerWAE(nn.Module):
    def __init__(self, model_type, isPreTrain, d_latent, device):
        super().__init__()

        """
        Initialize WAE model
        
        Args:
            encoder_config (dictionary): encoder transformer's configuration
            d_latent (int): latent dimension size
            device (torch.device): 
        Returns:
            log_prob (torch.Tensor): log probability of each word 
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
            z (torch.Tensor): sampled latent vector
        """
        self.d_latent = d_latent
        self.model_type = model_type
        self.isPreTrain = isPreTrain
        self.device = device

        if self.model_type == 'T5':
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            if self.isPreTrain:
                self.model1 = T5ForConditionalGeneration.from_pretrained('t5-small')
                self.model2 = T5ForConditionalGeneration.from_pretrained('t5-small')
            else:
                model_config = T5Config("t5-small")
                model_config.vocab_size = 32128
                self.model1 = T5ForConditionalGeneration(config=model_config)
                self.model2 = T5ForConditionalGeneration(config=model_config)
            # Encoder1 Setting
            self.encoder1_embedding = self.model1.encoder.embed_tokens
            self.encoder1_model = self.model1.encoder.block
            self.encoder1_final_layer_norm = self.model1.encoder.final_layer_norm
            self.encoder1_dropout = self.model1.encoder.dropout
            # Dimension Setting
            self.d_hidden = self.encoder1_embedding.embedding_dim
            # Encoder2 Setting
            self.encoder2_model = self.model2.get_encoder()
            # Decoder Setting
            self.decoder_model = self.model2.get_decoder()
            # Final Layer Setting
            self.vocab_size = self.model2.lm_head.out_features
            self.lm_head = self.model2.lm_head
        elif self.model_type == 'Bart':
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
            if self.isPreTrain:
                self.model1 = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
                self.model2 = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
            else:
                model_config = BartConfig('facebook/bart-base')
                model_config.vocab_size = 50265
                self.model1 = BartForConditionalGeneration(config=model_config)
                self.model2 = BartForConditionalGeneration(config=model_config)
            # Encoder1 Setting
            self.encoder1_model = self.model1.get_encoder()
            # Dimension Setting
            self.d_hidden = self.encoder1_model.embed_tokens.embedding_dim
            # Encoder2 Setting
            self.encoder2_model = self.model2.get_encoder()
            # Decoder Setting
            self.decoder_model = self.model2.get_decoder()
            # Final Layer Setting
            self.vocab_size = self.model2.lm_head.out_features
            self.lm_head = self.model2.lm_head
        elif self.model_type == 'BERT':
            # To Do
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            raise ValueError('not supported')

        # For latent mapping
        # self.hidden2latent = nn.Linear(self.d_hidden, self.d_latent)
        # self.latent2hidden = nn.Linear(self.d_latent, self.d_hidden)

    def forward(self, input_ids, attention_mask, token_type_ids=None):

        if self.model_type == 'T5':
            # Encoder1 Forward
            wae_enc_out = self.encoder1_embedding(input_ids)
            new_attention_mask = self.model1.get_extended_attention_mask(attention_mask, 
                                                                         attention_mask.shape, self.device)
            for i in range(len(self.encoder1_model)):
                wae_enc_out, _ = self.encoder1_model[i](hidden_states=wae_enc_out, 
                                                        attention_mask=new_attention_mask)

            wae_enc_out = self.encoder1_final_layer_norm(wae_enc_out)
            wae_enc_out = self.encoder1_dropout(wae_enc_out)

            # Encoder2 Forward
            wae_dec_out = self.encoder2_model(inputs_embeds=wae_enc_out, 
                                              attention_mask=attention_mask)
            wae_dec_out = wae_dec_out['last_hidden_state']

            # Decoder
            model_out = self.decoder_model(inputs_embeds=wae_enc_out, 
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=wae_dec_out,
                                           encoder_attention_mask=attention_mask)
            model_out = self.lm_head(model_out['last_hidden_state'])

            return wae_enc_out, wae_dec_out, model_out

        elif self.model_type == 'Bart':
            # Encoder1 Forward
            wae_enc_out = self.encoder1_model(input_ids=input_ids,
                                              attention_mask=attention_mask)
            wae_enc_out = wae_enc_out['last_hidden_state']

            # Encoder2 Forward
            wae_dec_out = self.encoder2_model(inputs_embeds=wae_enc_out, 
                                              attention_mask=attention_mask)
            wae_dec_out = wae_dec_out['last_hidden_state']

            # Decoder
            model_out = self.decoder_model(inputs_embeds=wae_enc_out, 
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=wae_dec_out,
                                           encoder_attention_mask=attention_mask)
            model_out = self.lm_head(model_out['last_hidden_state'])

            return wae_enc_out, wae_dec_out, model_out

class Discirminator_model(nn.Module):
    def __init__(self, model_type, isPreTrain, device, class_token='first_token'):
        super().__init__()

        self.model_type = model_type
        self.isPreTrain = isPreTrain
        self.class_token = class_token
        self.device = device

        if self.model_type == 'T5':
            if self.isPreTrain:
                self.D_model = T5EncoderModel.from_pretrained('t5-small')
            else:
                model_config = T5Config.from_pretrained("t5-small")
                self.D_model = T5EncoderModel(config=model_config)
            d_model = self.D_model.encoder.embed_tokens.embedding_dim
            self.linear = nn.Linear(d_model, 1)

    def forward(self, z):
        out = self.D_model(inputs_embeds=z)
        out = out['last_hidden_state']
        out = self.linear(out)

        if self.class_token == 'first_token':
            return out[:,0,:]
        elif self.class_token == 'mean_pooling':
            return out.mean(dim=1)
        elif self.class_token == 'last_token':
            return out[:,-1,:]
        else:
            raise Exception('Choose class_token in [first_token, mean_pooling, last_token]')