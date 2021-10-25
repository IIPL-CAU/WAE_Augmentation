import math
# Import PyTorch
import torch
import torch.nn as nn
# Import Huggingface
# T5
from transformers import T5ForConditionalGeneration, T5EncoderModel, T5Config, T5Tokenizer
# Bart
from transformers import BartTokenizer

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
            self.lm_head = self.model2.lm_head
            # Final Layer Setting
            self.vocab_size = self.model2.lm_head.out_features
        elif self.model_type == 'Bart':
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
            self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
            self.vocab_size = self.model.lm_head.out_features
        elif self.model_type == 'T5':
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
            self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
            self.vocab_size = self.model.lm_head.out_features
        elif self.model_type == 'Transformer':
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
            self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
            self.vocab_size = self.model2.lm_head.out_features

        # For latent mapping
        self.hidden2latent = nn.Linear(self.d_hidden, self.d_latent)
        self.latent2hidden = nn.Linear(self.d_latent, self.d_hidden)

    def forward(self, input_ids, attention_mask):

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

            # Decoder
            model_out = self.decoder_model(input_ids=input_ids, 
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=wae_dec_out['last_hidden_state'],
                                           encoder_attention_mask=attention_mask)
            model_out = self.lm_head(model_out['last_hidden_state'])

            return wae_enc_out, wae_dec_out, model_out

        elif self.model_type == 'Bart':
            # Encoder1 Forward
            wae_enc_out = self.encoder_embedding(input_ids)
            new_attention_mask = self.model.get_extended_attention_mask(attention_mask, 
                                                                        attention_mask.shape, self.device)
            for i in range(len(self.encoder_model)):
                wae_enc_out, _ = self.encoder_model[i](hidden_states=wae_enc_out, 
                                                       attention_mask=new_attention_mask)

            wae_enc_out = self.encoder_final_layer_norm(wae_enc_out)
            wae_enc_out = self.encoder_dropout(wae_enc_out)

            # Encoder2 Forward
            wae_dec_out = self.encoder2_model(inputs_embeds=wae_enc_out, 
                                              attention_mask=attention_mask)

            # Decoder
            model_out = self.decoder_model(input_ids=input_ids, 
                                           attention_mask=attention_mask,
                                           encoder_hidden_states=wae_dec_out,
                                           encoder_attention_mask=attention_mask)
            model_out = self.lm_head(model_out['last_hidden_state'])

            return wae_enc_out, wae_dec_out, model_out


def sample_z(args, n_sample=None, dim=None, sigma=None, template=None):
    if n_sample is None:
        n_sample = args.batch_size
    if dim is None:
        dim = args.d_model
    if sigma is None:
        sigma = math.sqrt(args.z_var)

    if template is not None:
        z = sigma*template.data.new(template.size()).normal_()
    else:
        z = sigma*torch.randn(n_sample, dim)

    return z

def im_kernel_sum(z1, z2, z_var, exclude_diag=True):
    r"""Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.
    Args:
        z1 (Tensor): batch of samples from a multivariate gaussian distribution \
            with scalar variance of z_var.
        z2 (Tensor): batch of samples from another multivariate gaussian distribution \
            with scalar variance of z_var.
        exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2*z_dim*z_var

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C/(1e-9+C+(z11-z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()
    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum

def mmd(z_tilde, z, z_var):
    r"""Calculate maximum mean discrepancy described in the WAE paper.
    Args:
        z_tilde (Tensor): samples from deterministic non-random encoder Q(Z|X).
            2D Tensor(batch_size x dimension).
        z (Tensor): samples from prior distributions. same shape with z_tilde.
        z_var (Number): scalar variance of isotropic gaussian prior P(Z).
    """
    assert z_tilde.size() == z.size()
    assert z.ndimension() == 2

    n = z.size(0)
    out = im_kernel_sum(z, z, z_var, exclude_diag=True).div(n*(n-1)) + \
          im_kernel_sum(z_tilde, z_tilde, z_var, exclude_diag=True).div(n*(n-1)) + \
          -im_kernel_sum(z, z_tilde, z_var, exclude_diag=False).div(n*n).mul(2)

    return out

def log_density_igaussian(z, z_var):
    """Calculate log density of zero-mean isotropic gaussian distribution given z and z_var."""
    assert z.ndimension() == 2
    assert z_var > 0

    z_dim = z.size(1)

    return -(z_dim/2)*math.log(2*math.pi*z_var) + z.pow(2).sum(1).div(-2*z_var)

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