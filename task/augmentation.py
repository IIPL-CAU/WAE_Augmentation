# Import Modules
import os
import time
import pickle
import logging
from tqdm import tqdm
# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
# Import Custom Modules
from model.dataset import CustomDataset, PadCollate
from model.wae import TransformerWAE, mmd, sample_z
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log

def augmenting(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

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
    with open(f'{args.preprocess_path}/{args.dataset}_{args.tokenizer}_preprocessed.pkl', 'rb') as f:
        data_ = pickle.load(f)
        train_input_ids = data_['train']['input_ids']
        train_attention_mask = data_['train']['attention_mask']
        train_label = data_['train']['label']
        valid_input_ids = data_['valid']['input_ids']
        valid_attention_mask = data_['valid']['attention_mask']
        valid_label = data_['valid']['label']
        if args.tokenizer == 'T5':
            train_token_type_ids = None
            valid_token_type_ids = None
        else:
            train_token_type_ids = data_['train']['token_type_ids']
            valid_token_type_ids = data_['valid']['token_type_ids']
        del data_

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(tokenizer=args.tokenizer, input_ids_list=train_input_ids,
                               label_list=train_label, attention_mask_list=train_attention_mask,
                               token_type_ids_list=train_token_type_ids, min_len=4, max_len=512),
        'valid': CustomDataset(tokenizer=args.tokenizer, input_ids_list=train_input_ids,
                               label_list=train_label, attention_mask_list=train_attention_mask,
                               token_type_ids_list=valid_token_type_ids, min_len=4, max_len=512)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(args.tokenizer), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], collate_fn=PadCollate(args.tokenizer), drop_last=False,
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
    model = TransformerWAE(d_hidden=args.d_model, d_latent=args.d_model)
    model = model.train()
    model = model.to(device)

    # 2) Optimizer setting
    optimizer = optimizer_select(model, args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)
    scaler = GradScaler()

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint.pth.tar'), map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        model = model.train()
        model = model.to(device)
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_val_acc = 0

    write_log(logger, 'Train start!')

    for epoch in range(start_epoch, args.num_epochs):

        # Train setting
        start_time_e = time.time()
        model = model.train()

        for i, (input_ids, attention_mask, label) in enumerate(tqdm(dataloader_dict['train'])):

            #===================================#
            #============Train Epoch============#
            #===================================#

            # Optimizer setting
            optimizer.zero_grad()

            # Input, output setting
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            # Model
            enc_out, z_tilde, ae_hidden, dec_out = model(input_ids, attention_mask)
            z = sample_z(args=args, template=z_tilde)

            # Loss calculate
            recon_loss = F.cross_entropy(dec_out.view(-1, dec_out.size(-1)), 
                                         input_ids.contiguous().view(-1), 
                                         ignore_index=args.pad_idx)
            mmd_loss = mmd(z_tilde.view(args.batch_size, -1), z.view(args.batch_size, -1), 
                           z_var=args.z_var)
            total_loss = recon_loss + args.loss_lambda*mmd_loss
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            if args.scheduler in ['constant', 'warmup']:
                scheduler.step()
            if args.scheduler == 'reduce_train':
                scheduler.step(mlm_loss)

            # Print loss value only training
            acc = sum(input_ids.view(-1) == dec_out.view(-1, dec_out.size(-1)).max(dim=1)[1]) / len(input_ids.view(-1))
            acc = acc.item() * 100
            if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train'])-1:
                batch_log = "[Epoch:%d][%d/%d] train_recon_loss:%2.3f | train_mmd_loss:%2.3f | train_acc:%02.2f | learning_rate:%3.6f | spend_time:%3.2fmin" \
                        % (epoch+1, i+1, len(dataloader_dict['train']), 
                        recon_loss.item(), mmd_loss.item(), acc, optimizer.param_groups[0]['lr'], 
                        (time.time() - start_time_e) / 60)
                write_log(logger, batch_log)
                freq = 0
            freq += 1

        #===================================#
        #=========Validation Epoch==========#
        #===================================#

        # Validation setting
        model = model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for i, (input_ids, attention_mask, label) in enumerate(tqdm(dataloader_dict['valid'])):

                # Input, output setting
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)

                # Model
                enc_out, z_tilde, ae_hidden, dec_out = model(input_ids, attention_mask)

                # Loss calculate
                recon_loss = F.cross_entropy(dec_out.view(-1, dec_out.size(-1)), 
                                            input_ids.contiguous().view(-1), 
                                            ignore_index=args.pad_idx)
                mmd_loss = mmd(z_tilde.view(args.batch_size, -1), z.view(args.batch_size, -1), 
                            z_var=args.z_var)
                total_loss = recon_loss + args.loss_lambda*mmd_loss

                # Print loss value only training
                acc = sum(input_ids.view(-1) == dec_out.view(-1, dec_out.size(-1)).max(dim=1)[1]) / len(input_ids.view(-1))
                acc = acc.item() * 100
                val_loss += total_loss.item()
                val_acc += acc

        # Show Example
        original_sent = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        generated_sent = model.tokenizer.batch_decode(dec_out.max(dim=2)[1], skip_special_tokens=True)
        write_log(logger, 'Original Sentence:')
        write_log(logger, original_sent)
        write_log(logger, 'Generated Sentence:')
        write_log(logger, generated_sent)

        val_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Validation Loss: %3.3f' % val_loss)
        write_log(logger, 'Validation Accuracy: %3.2f%%' % val_acc)

        if val_acc > best_val_acc:
            write_log(logger, 'Checkpoint saving...')
            # Checkpoint path setting
            if not os.path.exists(args.vit_save_path):
                os.mkdir(args.vit_save_path)
            # Save
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }, os.path.join(args.save_path, f'checkpoint.pth.tar'))
            best_val_acc = val_acc
            best_epoch = epoch
        else:
            else_log = f'Still {best_epoch} epoch accuracy({round(best_val_acc, 2)})% is better...'
            write_log(logger, else_log)