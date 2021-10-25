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
from model.wae.model import TransformerWAE, Discirminator_model
from model.wae.loss import mmd, sample_z, log_density_igaussian
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log

def augmenting(args):

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
    with open(f'{args.preprocess_path}/{args.dataset}_{args.model_type}_preprocessed.pkl', 'rb') as f:
        data_ = pickle.load(f)
        train_input_ids = data_['train']['input_ids']
        train_attention_mask = data_['train']['attention_mask']
        train_label = data_['train']['label']
        valid_input_ids = data_['valid']['input_ids']
        valid_attention_mask = data_['valid']['attention_mask']
        valid_label = data_['valid']['label']
        if args.tokenizer in ['T5', 'Bart']:
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
        'valid': CustomDataset(tokenizer=args.tokenizer, input_ids_list=valid_input_ids,
                               label_list=valid_label, attention_mask_list=valid_attention_mask,
                               token_type_ids_list=valid_token_type_ids, min_len=4, max_len=512)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(args.tokenizer), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], collate_fn=PadCollate(args.tokenizer), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)
    }

    print_text = f"Total number of trainingsets iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}"
    write_log(logger, print_text)

    #===================================#
    #============Model Load=============#
    #===================================#

    # 1) Model initiating
    write_log(logger, "Instantiating models...")
    model = TransformerWAE(model_type=args.model_type, isPreTrain=args.PLM_use,
                           d_latent=args.d_latent, device=device)
    model = model.train()
    model = model.to(device)
    
    # 1-1) Discriminator for WAE-GAN Mode
    if args.WAE_loss == 'gan':
        D_model = Discirminator_model(model_type=args.model_type, isPreTrain=args.PLM_use,
                                      device=device, class_token='first_token')
        D_model = D_model.train()
        D_model = D_model.to(device)
        optimizer_d = optimizer_select(D_model, args)
        scaler_d = GradScaler()

    # 2) Optimizer setting
    optimizer = optimizer_select(model, args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)
    scaler = GradScaler()

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        save_name = f'{args.dataset}_{args.model_type}_wae_checkpoint.pth.tar'
        checkpoint = torch.load(os.path.join(args.save_path, save_name), map_location='cpu')
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

        for i, input_ in enumerate(tqdm(dataloader_dict['train'], 
                                                               bar_format='{l_bar}{bar:30}{r_bar}{bar:-30}')):

            #===================================#
            #============Train Epoch============#
            #===================================#

            # Optimizer setting
            optimizer.zero_grad()

            # Input, output setting
            if len(input_) == 3:
                input_ids = input_[0].to(device, non_blocking=True)
                attention_mask = input_[1].to(device, non_blocking=True)
                token_type_ids = None
            if len(input_) == 4:
                input_ids = input_[0].to(device, non_blocking=True)
                token_type_ids = input_[1].to(device, non_blocking=True)
                attention_mask = input_[2].to(device, non_blocking=True)

            # Model
            wae_enc_out, wae_dec_out, model_out = model(input_ids, attention_mask, token_type_ids)
            z = sample_z(args=args, template=wae_enc_out)

            # Loss calculate
            recon_loss = F.cross_entropy(model_out.view(-1, model_out.size(-1)), 
                                         input_ids.contiguous().view(-1), 
                                         ignore_index=args.pad_idx)
            if args.WAE_loss == 'mmd':
                mmd_loss = mmd(wae_enc_out.view(args.batch_size, -1), 
                               z.view(args.batch_size, -1), 
                               z_var=args.z_var)
                total_loss = recon_loss + args.loss_lambda*mmd_loss
            elif args.WAE_loss == 'gan':
                # Discriminator Model Forward
                D_z = D_model(z)
                D_z_tilde = D_model(wae_enc_out)

                # Pre-setting
                log_p_z = log_density_igaussian(z.view(args.batch_size, -1), args.z_var).view(-1, 1)
                ones = torch.ones(args.batch_size, 1).to(device)
                zeros = torch.zeros(args.batch_size, 1).to(device)

                # Loss Calculate
                D_loss = F.binary_cross_entropy_with_logits(D_z+log_p_z, ones) + \
                         F.binary_cross_entropy_with_logits(D_z_tilde+log_p_z, zeros)
                total_D_loss = args.loss_lambda*D_loss

                # Loss Back-propagation
                optimizer_d.zero_grad()
                scaler_d.scale(total_D_loss).backward(retain_graph=True)
                scaler_d.unscale_(optimizer_d)
                clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler_d.step(optimizer_d)
                scaler_d.update()

                # Total Loss
                Q_loss = F.binary_cross_entropy_with_logits(D_z_tilde+log_p_z, ones)
                total_loss = recon_loss + args.loss_lambda*Q_loss

            # Back-propagation
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            if args.scheduler in ['constant', 'warmup']:
                scheduler.step()
            if args.scheduler == 'reduce_train':
                scheduler.step(total_loss)

            # Print loss value only training
            acc = sum(input_ids.view(-1) == model_out.view(-1, model_out.size(-1)).max(dim=1)[1]) / len(input_ids.view(-1))
            acc = acc.item() * 100
            if i == 0 or freq == args.print_freq-1 or i==len(dataloader_dict['train'])-1:
                if args.WAE_loss == 'mmd':
                    batch_log = "[Epoch:%d][%d/%d] train_recon_loss:%2.3f | train_mmd_loss:%2.3f | train_acc:%02.2f | learning_rate:%3.6f | spend_time:%3.2fmin" \
                            % (epoch+1, i+1, len(dataloader_dict['train']), 
                            recon_loss.item(), mmd_loss.item(), acc, optimizer.param_groups[0]['lr'], 
                            (time.time() - start_time_e) / 60)
                elif args.WAE_loss == 'gan':
                    batch_log = "[Epoch:%d][%d/%d] train_recon_loss:%2.3f | train_gan_loss:%2.3f | train_acc:%02.2f | learning_rate:%3.6f | spend_time:%3.2fmin" \
                            % (epoch+1, i+1, len(dataloader_dict['train']), 
                            recon_loss.item(), total_D_loss.item(), acc, optimizer.param_groups[0]['lr'], 
                            (time.time() - start_time_e) / 60)
                write_log(logger, batch_log)
                freq = -1
            freq += 1

        #===================================#
        #=========Validation Epoch==========#
        #===================================#

        # Validation setting
        model = model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for i, (input_ids, attention_mask, label) in enumerate(tqdm(dataloader_dict['valid'],
                                                                        bar_format='{l_bar}{bar:30}')):

                # Input, output setting
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)

                # Model
                wae_enc_out, wae_dec_out, model_out = model(input_ids, attention_mask)

                # Loss calculate
                recon_loss = F.cross_entropy(model_out.view(-1, model_out.size(-1)), 
                                            input_ids.contiguous().view(-1), 
                                            ignore_index=args.pad_idx)
                mmd_loss = mmd(wae_enc_out.view(args.batch_size, -1), 
                               z.view(args.batch_size, -1), 
                               z_var=args.z_var)
                total_loss = recon_loss + args.loss_lambda*mmd_loss

                # Print loss value only training
                acc = sum(input_ids.view(-1) == model_out.view(-1, model_out.size(-1)).max(dim=1)[1]) / len(input_ids.view(-1))
                acc = acc.item() * 100
                val_loss += total_loss.item()
                val_acc += acc

        # Show Example
        original_sent = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        generated_sent = model.tokenizer.batch_decode(model_out.max(dim=2)[1], skip_special_tokens=True)
        with open(f'example_{args.dataset}_{args.model_type}_{args.WAE_loss}.txt', 'a') as f:
            f.write(f'epoch: {epoch} \n')
            f.write('Original Sentence\n')
            f.write(f'{original_sent}\n')
            f.write('Generated Sentence\n')
            f.write(f'{generated_sent}\n')
        write_log(logger, 'Original Sentence:')
        write_log(logger, original_sent[0])
        write_log(logger, 'Generated Sentence:')
        write_log(logger, generated_sent[0])

        val_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Validation Loss: %3.3f' % val_loss)
        write_log(logger, 'Validation Accuracy: %3.2f%%' % val_acc)

        if val_acc > best_val_acc:
            write_log(logger, 'Checkpoint saving...')
            # Checkpoint path setting
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            # Save
            save_name = f'{args.dataset}_{args.model_type}_wae_checkpoint.pth.tar'
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }, os.path.join(args.save_path, save_name))
            best_val_acc = val_acc
            best_epoch = epoch
            if args.WAE_loss == 'gan':
                save_name_d = f'{args.dataset}_{args.model_type}_wae_discriminator_checkpoint.pth.tar'
                torch.save({
                    'epoch': epoch,
                    'model': D_model.state_dict(),
                    'optimizer': optimizer_d.state_dict(),
                    'scaler': scaler_d.state_dict()
                }, os.path.join(args.save_path, save_name_d))
        else:
            else_log = f'Still {best_epoch} epoch accuracy({round(best_val_acc, 2)})% is better...'
            write_log(logger, else_log)