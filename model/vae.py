import random

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F


class GeneratorVAE(nn.Module):
    def __init__(self, spm_model, batch_size, vocab_size=12004, embed_size=512, hidden_size=256, latent_size=32, num_layers=1,
                device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize model
        
        Args:
            spm_model (SPM): trained spm model
            batch_size (int): batch size
            vocab_size (int): vocabulary size
            embed_size (int): size of embedded word
            hidden_size (int): hidden size for gru
            latent_size (int): size of latent vector z
            num_layers (int): number of gru layers
            device (str): device to use
        """

        super(GeneratorVAE, self).__init__()
        self.spm_model = spm_model
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.device = device

        self.text_embed = nn.Embedding(vocab_size, embed_size)
        self.encoder = EncoderGRU(batch_size, embed_size, hidden_size, latent_size, num_layers, device=device)
        self.decoder = DecoderGRU(batch_size, embed_size, hidden_size, latent_size, vocab_size, spm_model=self.spm_model,
                                  embed_layer=self.text_embed, num_layers=self.num_layers, device=device)

    def forward(self, input_text, non_pad_length):
        """Forward pass of VAE model

        Args:
            input_text (torch.Tensor): encoded input text tensor
            non_pad_length (torch.Tensor): tensor of non-padding length
        Returns:
            log_prob (torch.Tensor): log probability of each word 
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
            z (torch.Tensor): sampled latent vector
        """
        input_embedding = self.text_embed(input_text)

        z, mean, log_var = self.encoder(input_embedding, non_pad_length)
        log_prob = self.decoder(z, input_embedding, non_pad_length)

        return log_prob, mean, log_var, z

    def get_embedding(self, input_text):
        """Get the embedding of input text, used for test sequence

        Args:
            input_text (torch.Tensor): encoded input text tensor
        Returns:
            torch.Tensor: embedding of input text
        """
        return self.text_embed(input_text)


class EncoderGRU(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, latent_size, num_layers=1, 
                device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize model

        Args: 
            batch_size (int): batch size
            input_size (int): size of input word embedding vector
            hidden_size (int): hidden size for gru
            latent_size (int): size of latent vector
            num_layers (int): number of gru layers
            device (str): device to uses
        """
        super(EncoderGRU, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.device = device

        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear_mean = nn.Linear(hidden_size, latent_size)
        self.linear_logvar = nn.Linear(hidden_size, latent_size)
    
    def encode(self, input_embedding, length):
        """Encode given input embedding of text to mean and log variance of latent vector

        Args:
            input_embedding (torch.Tensor): input embedding of text
            length (torch.Tensor): non-pad length of input text
        Return:
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
        """
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_embedding = input_embedding[sorted_idx]

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.rnn(packed_input) # hidden: (num_layers, batch_size, hidden_size)

        mean = self.linear_mean(hidden) # mean: (num_layers, batch_size, latent_size)
        log_var = self.linear_logvar(hidden) # log_var: (num_layers, batch_size, latent_size)

        return mean, log_var

    def reparameterize(self, mean, log_var):
        """Sample latent vector z from mean and log variance
        
        Args:
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
        Return:
            z (torch.Tensor): sampled latent vector
        """
        batch_size = mean.size(0)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = eps * std + mean

        return z

    def forward(self, input_embedding, length):
        """Forward pass of encoder
        
        Args:
            input_embedding (torch.Tensor): input embedding of text
            length (torch.Tensor): non-pad length of input text
        Return:
            z (torch.Tensor): sampled latent vector
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
        """

        mean, log_var = self.encode(input_embedding, length)
        z = self.reparameterize(mean, log_var)

        return z, mean, log_var


class DecoderGRU(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, latent_size, vocab_size, spm_model,
                embed_layer, num_layers=1, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize model

        Args:
            batch_size (int): batch size
            input_size (int): size of input word embedding vector
            hidden_size (int): hidden size for gru
            latent_size (int): size of latent vector
            vocab_size (int): size of vocabulary
            spm_model: sentencepiece Model
            embed_layer (nn.Module): torch module for embedding layer
            num_layers (int): number of gru layers
            device (str): device to uses
        """
        super(DecoderGRU, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.spm_model = spm_model
        self.num_layers = num_layers
        self.device = device

        self.text_embed = embed_layer
        self.linear_hidden = nn.Linear(latent_size, hidden_size) # Linear layer, from z to hidden state of rnn
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear_vocab = nn.Linear(num_layers * hidden_size, vocab_size)
        self.activation_vocab = nn.ReLU()

    def forward(self, z, input_embedding, length):
        """Forward pass of decoder

        Args:
            z (torch.Tensor): sampled latent vector by encoder 
            input_embedding (torch.Tensor): input embedding of text
            length (torch.Tensor): non-pad length of input text
        Return:
            log_prob (torch.Tensor): log probability of generated text
        """
        hidden = self.linear_hidden(z) # hidden: (num_layers, batch_size, hidden_size)

        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_embedding = input_embedding[sorted_idx]

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        output, _ = self.rnn(packed_input, hidden) # output: (batch_size, seq_len, hidden_size) / hidden: (num_layers, batch_size, hidden_size)

        padded_output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True, total_length=input_embedding.size(1)) # output: (batch_size, seq_len, vocab_size)
        padded_output = padded_output.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_output = padded_output[reversed_idx] # output: (batch_size, seq_len, vocab_size)
    
        logits = self.linear_vocab(padded_output) # logits: (batch_size, seq_len, vocab_size)

        log_prob = F.log_softmax(logits, dim=2) # log_prob: (batch_size, seq_len, vocab_size)

        return log_prob

    def decode(self, z, input_embedding):
        """Generate sentence for test sequence using given z
        
        Args:
            z (torch.Tensor): sampled latent vector by encoder
            input_embedding (torch.Tensor): input embedding of text
        Return:
            outputs (torch.Tensor): integer index of generated text
            outputs_sentence (list): list of string of generated text
        """
        hidden = self.linear_hidden(z)
        outputs = torch.zeros((input_embedding.size(0), input_embedding.size(1))).long().to(self.device) # outputs: (batch_size, seq_len)
        input = input_embedding[:, 0, :]
        input = input.unsqueeze(1) # input: (batch_size, 1, embed_size)
        for i in range(0, input_embedding.size(1)):
            rnn_output, hidden = self.rnn(input, hidden) # rnn_output: (batch_size, 1, hidden_size) / hidden: (num_layers, batch_size, hidden_size)
            logit = self.linear_vocab(rnn_output) # logit: (batch_size, 1, vocab_size)

            input_idx = self.sample_word_from_dist(logit) # input_idx: (batch_size, 1)
            outputs[:, i] = input_idx.squeeze()
            input = self.text_embed(input_idx) # input: (batch_size, 1, embed_size)

        # get eos_token
        outputs_sentence = []
        for each_line in outputs:
            for i in range(0, len(each_line)):
                if each_line[i] == self.spm_model.eos_id():
                    each_line = each_line[:i]
                    break
            outputs_sentence.append(self.spm_model.DecodeIds(each_line.tolist()))

        return outputs, outputs_sentence

    def sample_word_from_dist(self, log_prob, k=10):
        """Sample a word from given probablity distribution.

        Args:
            log_prob (torch.Tensor): probability distribution of vocabulary
            k (int): number of top-k words to be sampled as candidate
        Return:
            input_idx (torch.Tensor): integer index of sampled word
        """
        batch_size = log_prob.size(0)

        topk_prob, topk_indices = torch.topk(log_prob, k, dim=2) # topk_prob: (batch_size, seq_len, k)

        sampled = torch.multinomial(topk_prob.squeeze(), num_samples=1).squeeze()

        word_idx = torch.zeros(batch_size, 1).long().to(self.device) 
        for i in range(0, word_idx.size(0)):
            word_idx[i] = topk_indices[i][0][sampled[i]]

        return word_idx