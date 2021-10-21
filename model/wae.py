import math
# Import PyTorch
import torch
import torch.nn as nn
# Import Huggingface
from transformers import T5ForConditionalGeneration

class TransformerWAE(nn.Module):
    def __init__(self, d_hidden, d_latent):
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
        self.d_hidden = d_hidden

        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.vocab_size = self.model.lm_head.out_features

        self.hidden2latent = nn.Linear(self.d_hidden, self.d_latent)
        self.latent2hidden = nn.Linear(self.d_latent, self.d_hidden)

    def forward(self, input_ids, attention_mask):

        emb_ = self.model.shared(input_ids)
        enc_out = self.model.encoder(inputs_embeds = emb_, attention_mask = attention_mask)

        # Wasserstein Auto-encoder
        z = self.hidden2latent(enc_out['last_hidden_state'])
        ae_hidden = self.latent2hidden(z)

        dec_out = self.model.decoder(input_ids=input_ids, 
                                     attention_mask=attention_mask,
                                     encoder_hidden_states=ae_hidden,
                                     encoder_attention_mask=attention_mask)
        dec_out = self.model.lm_head(dec_out['last_hidden_state'])

        return enc_out, z, ae_hidden, dec_out

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