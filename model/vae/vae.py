# Import PyTorch
import torch
import torch.nn as nn
# Import Huggingface
from transformers import EncoderDecoderModel, EncoderDecoderConfig

class GaussianKLLoss(nn.Module):
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu1, logvar1, mu2, logvar2):
        numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
        fraction = torch.div(numerator, (logvar2.exp()))
        kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1, dim=1)
        return kl.mean(dim=0)

class SentenceVAE(nn.Module):
    def __init__(self, encoder_config, decoder_config, d_latent, device):
        super().__init__()

        """
        Initialize VAE model
        
        Args:
            encoder_config (dictionary): encoder transformer's configuration
            encoder_config (dictionary): decoder transformer's configuration
            d_latent (int): latent dimension size
            device (torch.device): 
        Returns:
            log_prob (torch.Tensor): log probability of each word 
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
            z (torch.Tensor): sampled latent vector
        """

        self.d_latent = d_latent
        self.device = device

        self.d_hidden = encoder_config.hidden_size

        EncoderDecoderConfig = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.vocab_size = encoder_config.vocab_size

        self.seq2seq_bert = EncoderDecoderModel(config=EncoderDecoderConfig)
        self.seq2seq_bert_encoder = self.seq2seq_bert.encoder
        self.seq2seq_bert_decoder = self.seq2seq_bert.decoder

        self.hidden2mean = nn.Linear(self.d_hidden, self.d_latent)
        self.hidden2logv = nn.Linear(self.d_hidden, self.d_latent)
        self.latent2hidden = nn.Linear(self.d_latent, self.d_hidden)

    def reparameterization(self, enc_out):
        """
        Re-parameterization trick

        Args:
            enc_out (): 
        """
        mu = self.hidden2mean(enc_out.last_hidden_state)
        logv = self.hidden2logv(enc_out.last_hidden_state)
        std = torch.exp(0.5 * logv)

        z = torch.randn(enc_out.last_hidden_state.size(), device=self.device)
        z = z * std + mean

        return mu, logv, z

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
        batch_size = input_ids.size(0)
        encoder_input_ids = input_ids.clone()
        decoder_input_ids = input_ids.clone()

        # Encoder forward pass
        enc_out = self.seq2seq_bert_encoder(input_ids=encoder_input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)

        # Re-parameterization
        mu, logv, z = self.reparameterization(enc_out)

        # Decoder input processing
        hidden = self.latent2hidden(z)
        decoder_input_ids = self.embedding_dropout(decoder_input_ids)

        # Decoder forward pass
        outputs = self.seq2seq_bert_decoder(input_ids=decoder_input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids,
                                            encoder_hidden_states=enc_out.last_hidden_state,
                                            encoder_attention_mask=attention_mask)

        return outputs, mu, logv, z