# Import PyTorch
import torch
import torch.nn as nn
# Import Huggingface
from transformers import EncoderDecoderConfig, EncoderDecoderModel

class SentenceVAE(nn.Module):
    def __init__(self, encoder_config, decoder_config, d_latent, max_len):
        super().__init__()

        """
        Initialize VAE model
        
        Args:
            encoder_config (dictionary): encoder transformer's configuration
            encoder_config (dictionary): decoder transformer's configuration
            d_latent (int): latent dimension size
            max_len (int): max length of input sequence
        Returns:
            log_prob (torch.Tensor): log probability of each word 
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
            z (torch.Tensor): sampled latent vector
        """

        self.d_latent = d_latent
        self.max_len = max_len

        self.d_hidden = encoder_config.hidden_size

        EncoderDecoderConfig = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.seq2seq_bert = EncoderDecoderModel(config=EncoderDecoderConfig)
        self.seq2seq_bert_encoder = self.seq2seq_bert.encoder
        self.seq2seq_bert_decoder = self.seq2seq_bert.decoder

        self.hidden2mean = nn.Linear(self.d_hidden, self.d_latent)
        self.hidden2logv = nn.Linear(self.d_hidden, self.d_latent)
        self.latent2hidden = nn.Linear(self.d_latent, self.d_hidden)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Forward pass of VAE model

        Args:
            input_ids (torch.Tensor): encoded input tensor
            attention_mask (torch.Tensor): 
        Returns:
            log_prob (torch.Tensor): log probability of each word 
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
            z (torch.Tensor): sampled latent vector
        """
        batch_size = input_sequence.size(0)
        encoder_input_sequence = input_sequence.clone()
        decoder_input_sequence = input_sequence.clone()

        # Encoder forward pass
        enc_out = self.seq2seq_bert_encoder(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)

        # Flatten

        # Re-parameterization
        mu = self.hidden2mean(enc_out.last_hidden_state)
        logv = self.hidden2logv(enc_out.last_hidden_state)
        std = torch.exp(0.5 * logv)

        z = torch.randn([batch_size, self.d_latent])
        z = z * std + mean

        # Decoder input processing
        hidden = self.latent2hidden(z)
        input_sequence = self.embedding_dropout(input_sequence)

        # Decoder forward pass
        outputs = self.seq2seq_bert_decoder(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids,
                                            encoder_hidden_states=enc_out.last_hidden_state,
                                            encoder_attention_mask=attention_mask)