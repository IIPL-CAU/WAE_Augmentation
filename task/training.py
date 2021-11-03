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
from model.classification.dataset import CustomDataset, PadCollate
from model.classification.model import Classifier
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log

def training(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logger setting
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, "Training Start")

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data Open
    processed_path = os.path.join(args.preprocess_path, 
                                    f'{args.dataset}_{args.cls_tokenizer}_valid_ratio_{args.valid_split_ratio}_preprocessed.pkl')
    with open(processed_path, 'rb') as f:
        data_ = pickle.load(f)
        train_input_ids = data_['train']['input_ids']
        train_attention_mask = data_['train']['attention_mask']
        train_label = data_['train']['label']
        valid_input_ids = data_['valid']['input_ids']
        valid_attention_mask = data_['valid']['attention_mask']
        valid_label = data_['valid']['label']
        if args.cls_tokenizer in ['T5', 'Bart']:
            train_token_type_ids = None
            valid_token_type_ids = None
        else:
            train_token_type_ids = data_['train']['token_type_ids']
            valid_token_type_ids = data_['valid']['token_type_ids']
        del data_

    # 2) Augmented Data Open
    if args.train_with_augmentation:
        data_name = f'{args.dataset}_{args.aug_model_type}_aug_preprocessed.pkl'
        with open(os.path.join(args.preprocess_path, data_name), 'rb') as f:
            data_ = pickle.load(f)
            train_input_ids = train_input_ids + data_['augmented']['input_ids']
            train_attention_mask = train_attention_mask + data_['augmented']['attention_mask']
            train_label = train_label.tolist() + data_['augmented']['label']
            if args.cls_tokenizer in ['T5', 'Bart']:
                train_token_type_ids = None
            else:
                train_token_type_ids = train_token_type_ids + data_['augmented']['token_type_ids']
            del data_

    # 3) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(tokenizer=args.cls_tokenizer, 
                               input_ids_list=train_input_ids,
                               label_list=train_label, 
                               attention_mask_list=train_attention_mask,
                               token_type_ids_list=train_token_type_ids,
                               min_len=args.min_len, max_len=args.max_len),
        'valid': CustomDataset(tokenizer=args.cls_tokenizer, 
                               input_ids_list=valid_input_ids,
                               label_list=valid_label, 
                               attention_mask_list=valid_attention_mask,
                               token_type_ids_list=valid_token_type_ids, 
                               min_len=args.min_len, max_len=args.max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(args.cls_tokenizer), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], collate_fn=PadCollate(args.cls_tokenizer), drop_last=True,
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
    # if args.cls_model_type in ['RNN', 'CNN']:
    #     model_config = {
    #         'vocab_size': 
    #     }
    model = Classifier(model_type=args.cls_model_type, isPreTrain=args.cls_PLM_use,
                       num_class=len(set(train_label)))
    model = model.train()
    model = model.to(device)

    # 2) Optimizer setting
    optimizer = optimizer_select(model, args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        save_name = f'{args.dataset}_{args.model_type}_cls_checkpoint.pth.tar'
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

    best_val_loss = 1e+10

    write_log(logger, 'Train start!')

    for epoch in range(start_epoch, args.num_epochs):

        # Train setting
        start_time_e = time.time()
        model = model.train()

        for i, input_ in enumerate(tqdm(dataloader_dict['train'], 
                                   bar_format='{percentage:3.2f}%|{bar:50}{r_bar}')):

            #===================================#
            #============Train Epoch============#
            #===================================#

            # Optimizer setting
            optimizer.zero_grad()

            # Input, output setting
            if len(input_) == 3:
                input_ids = input_[0].to(device, non_blocking=True)
                attention_mask = input_[1].to(device, non_blocking=True)
                labels = input_[2].to(device, non_blocking=True)
                token_type_ids = None
            if len(input_) == 4:
                input_ids = input_[0].to(device, non_blocking=True)
                token_type_ids = input_[1].to(device, non_blocking=True)
                attention_mask = input_[2].to(device, non_blocking=True)
                labels = input_[3].to(device, non_blocking=True)

            # Model
            out = model(input_ids, attention_mask, token_type_ids)

            # Loss calculate
            loss = criterion(out, labels)

            # Back-propagation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            if args.scheduler in ['constant', 'warmup']:
                scheduler.step()
            if args.scheduler == 'reduce_train':
                scheduler.step(loss)

            # Print loss value only training
            acc = sum(labels == out.max(dim=1)[1]) / len(labels)
            acc = acc.item() * 100
            if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train'])-1:
                batch_log = "[Epoch:%d][%d/%d] train_loss:%2.3f | train_acc:%02.2f | learning_rate:%3.6f | spend_time:%3.2fmin" \
                        % (epoch+1, i+1, len(dataloader_dict['train']), 
                        loss.item(), acc, optimizer.param_groups[0]['lr'], 
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
            for i, input_ in enumerate(tqdm(dataloader_dict['valid'],
                                       bar_format='{percentage:3.2f}%|{bar:50}{r_bar}')):

                # Input, output setting
                if len(input_) == 3:
                    input_ids = input_[0].to(device, non_blocking=True)
                    attention_mask = input_[1].to(device, non_blocking=True)
                    labels = input_[2].to(device, non_blocking=True)
                    token_type_ids = None
                if len(input_) == 4:
                    input_ids = input_[0].to(device, non_blocking=True)
                    token_type_ids = input_[1].to(device, non_blocking=True)
                    attention_mask = input_[2].to(device, non_blocking=True)
                    labels = input_[3].to(device, non_blocking=True)

                # Model
                out = model(input_ids, attention_mask, token_type_ids)

                # Loss calculate
                loss = criterion(out, labels)

                # Accuracy & Loss
                acc = sum(labels == out.max(dim=1)[1]) / len(labels)
                acc = acc.item() * 100
                val_loss += loss.item()
                val_acc += acc

        val_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Validation Loss: %3.3f' % val_loss)
        write_log(logger, 'Validation Accuracy: %3.2f%%' % val_acc)

        # Reduce Validation Scheduler
        if args.scheduler == 'reduce_valid':
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            write_log(logger, 'Checkpoint saving...')
            # Checkpoint path setting
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            # Save
            save_name = f'{args.dataset}_{args.cls_model_type}_aug_{args.train_with_augmentation}_cls_checkpoint.pth.tar'
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }, os.path.join(args.save_path, save_name))
            best_val_loss = val_loss
            best_epoch = epoch
        else:
            else_log = f'Still {best_epoch+1} epoch loss({round(best_val_loss, 2)}) is better...'
            write_log(logger, else_log)