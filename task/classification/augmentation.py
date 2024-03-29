# Import Modules
import os
import time
import pickle
import logging
import pandas as pd
from tqdm import tqdm
# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# Import Huggingface
from transformers import BertTokenizerFast
# Import Custom Modules
from model.wae.dataset import CustomDataset, PadCollate
from model.wae.model import TransformerWAE, Discirminator_model
from model.vae.model import TransformerVAE
from model.wae.loss import mmd, sample_z, log_density_igaussian
from utils import TqdmLoggingHandler, write_log

def augmentation(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logger setting
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, "Augmenting Start")

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    processed_path = os.path.join(args.preprocess_path, 
                                  f'{args.dataset}_{args.aug_tokenizer}_valid_ratio_{args.valid_split_ratio}_preprocessed.pkl')
    with open(processed_path, 'rb') as f:
        data_ = pickle.load(f)
        train_input_ids = data_['train']['input_ids']
        train_attention_mask = data_['train']['attention_mask']
        train_label = data_['train']['label']
        valid_input_ids = data_['valid']['input_ids']
        valid_attention_mask = data_['valid']['attention_mask']
        valid_label = data_['valid']['label']
        if args.aug_tokenizer in ['T5', 'Bart']:
            train_token_type_ids = None
            valid_token_type_ids = None
        else:
            train_token_type_ids = data_['train']['token_type_ids']
            valid_token_type_ids = data_['valid']['token_type_ids']
        del data_

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(tokenizer=args.aug_tokenizer, input_ids_list=train_input_ids,
                               label_list=train_label, attention_mask_list=train_attention_mask,
                               token_type_ids_list=train_token_type_ids, min_len=4, max_len=512)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(args.aug_tokenizer), drop_last=True,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }

    print_text = f"Total number of trainingsets iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}"
    write_log(logger, print_text)

    #===================================#
    #============Model Load=============#
    #===================================#

    # 1) Model initiating
    write_log(logger, "Instantiating models...")
    if args.ae_type == 'WAE':
        model = TransformerWAE(model_type=args.aug_model_type, decoder_type=args.WAE_decoder,
                               isPreTrain=args.aug_PLM_use, d_latent=args.d_latent, device=device)
    if args.ae_type == 'VAE':
        model = TransformerVAE(model_type=args.aug_model_type, isPreTrain=args.aug_PLM_use,
                            d_latent=args.d_latent, device=device)
    
    # 1-1) Discriminator for WAE-GAN Mode
    if args.WAE_loss == 'gan':
        D_model = Discirminator_model(model_type=args.aug_model_type, isPreTrain=args.aug_PLM_use,
                                      device=device, class_token='first_token')

    # 2) Model load
    save_name = f'{args.dataset}_{args.aug_model_type}_{args.ae_type.lower()}_PLM_{args.aug_PLM_use}_checkpoint.pth.tar'
    if args.WAE_decoder is not 'Transformer':
        save_name = f'{args.dataset}_{args.aug_model_type}_{args.ae_type.lower()}_PLM_{args.aug_PLM_use}_{args.WAE_decoder}_checkpoint.pth.tar'
    checkpoint = torch.load(os.path.join(args.save_path, save_name), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    model = model.to(device)
    del checkpoint

    # 2-1) Discriminator model load
    if args.WAE_loss == 'gan':
        save_name_d = f'{args.dataset}_{args.aug_model_type}_wae_discriminator_checkpoint.pth.tar'
        checkpoint_d = torch.load(os.path.join(args.save_path, save_name_d), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        D_model = D_model.eval()
        D_model = D_model.to(device)
        del checkpoint_d

    #===================================#
    #============Augmentation===========#
    #===================================#

    total_sentence_list = list()
    total_label_list = list()

    write_log(logger, 'Augmentation start!')

    with torch.no_grad():
        for i, input_ in enumerate(tqdm(dataloader_dict['train'],
                                   bar_format='{percentage:3.2f}%|{bar:50}{r_bar}')):

            # Input, output setting
            if len(input_) == 3:
                input_ids = input_[0].to(device, non_blocking=True)
                attention_mask = input_[1].to(device, non_blocking=True)
                label_list = input_[2].tolist()
                token_type_ids = None
            if len(input_) == 4:
                input_ids = input_[0].to(device, non_blocking=True)
                token_type_ids = input_[1].to(device, non_blocking=True)
                attention_mask = input_[2].to(device, non_blocking=True)
                label_list = input_[3].tolist()

            # Model
            if args.ae_type == 'WAE':
                wae_enc_out, wae_dec_out, model_out = model(input_ids, attention_mask)
            if args.ae_type == 'VAE':
                wae_enc_out, wae_dec_out, model_out, kl = model(input_ids, attention_mask)

            # Decode
            generated_sent = model.tokenizer.batch_decode(model_out.max(dim=2)[1], skip_special_tokens=True)

            # Append
            total_sentence_list.extend(generated_sent)
            total_label_list.extend(label_list)

    #===================================#
    #===============Save================#
    #===================================#

    # CSV Save
    if args.ae_type == 'WAE':
        data_name = f'{args.dataset}_{args.aug_model_type}_{args.WAE_loss}.csv'
    if args.ae_type == 'VAE':
        data_name = f'{args.dataset}_{args.aug_model_type}_{args.ae_type.lower()}.csv'
    aug_dat = pd.DataFrame({
        'description': total_sentence_list,
        'label': total_label_list
    })
    aug_dat.to_csv(os.path.join(args.augmentation_path, data_name))

    # Pickle Save
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    encoded_out = tokenizer(
        total_sentence_list,
        max_length=args.max_len,
        padding='max_length',
        truncation=True
    )
    encoded_out['label'] = total_label_list

    if args.ae_type == 'WAE':
        data_name = f'{args.dataset}_{args.aug_model_type}_aug_preprocessed.pkl'
    if args.ae_type == 'VAE':
        data_name = f'{args.dataset}_{args.aug_model_type}_aug_{args.ae_type.lower()}_preprocessed.pkl'
    with open(os.path.join(args.preprocess_path, data_name), 'wb') as f:
        pickle.dump({
            'augmented': {
                'input_ids': encoded_out['input_ids'],
                'attention_mask': encoded_out['attention_mask'],
                'token_type_ids': encoded_out['token_type_ids'],
                'label': encoded_out['label']
            }
        }, f)