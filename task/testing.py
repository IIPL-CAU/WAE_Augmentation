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
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
# Import Custom Modules
from model.wae.dataset import CustomDataset, PadCollate
from model.classification.cnn import ClassifierCNN
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log

def testing(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logger setting
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, "Classification Start")

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    with open(f'{args.preprocess_path}/{args.dataset}_{args.model_type}_preprocessed.pkl', 'rb') as f:
        data_ = pickle.load(f)
        train_input_ids = data_['train']['input_ids']
        test_input_ids = data_['test']['input_ids']
        test_attention_mask = data_['test']['attention_mask']
        test_label = data_['test']['label']
        if args.tokenizer in ['T5', 'Bart']:
            test_token_type_ids = None
        else:
            test_token_type_ids = data_['test']['token_type_ids']
        del data_

    vocab_size = 0
    for each_line in train_input_ids:
        max_id = max(each_line)
        if max_id > vocab_size:
            vocab_size = max_id

    # 2) Dataloader setting
    dataset_dict = {
        'test': CustomDataset(tokenizer=args.tokenizer, input_ids_list=test_input_ids,
                               label_list=test_label, attention_mask_list=test_attention_mask,
                               token_type_ids_list=test_token_type_ids, min_len=4, max_len=512)
    }
    dataloader_dict = {
        'test': DataLoader(dataset_dict['test'], collate_fn=PadCollate(args.tokenizer), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)
    }

    #===================================#
    #============Model Load=============#
    #===================================#

    # 1) Model initiating
    write_log(logger, "Instantiating models...")
    model = ClassifierCNN(tokenizer_type=args.model_type, vocab_size=vocab_size+1,
                          max_len=args.max_len, class_num=max(test_label)+1, device=device)

    # 2) Model load
    save_name = f'{args.dataset}_{args.classifier_model_type}_classifier_checkpoint.pth.tar'
    checkpoint = torch.load(os.path.join(args.save_path, save_name), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    model = model.to(device)
    del checkpoint

    #===================================#
    #==========Classification===========#
    #===================================#

    total_sentence_list = list()
    total_label_list = list()
    total_prediction_list = list()
    test_acc = 0

    write_log(logger, 'Classification test start!')

    with torch.no_grad():
        for i, input_ in enumerate(tqdm(dataloader_dict['test'],
                                   bar_format='{l_bar}{bar:30}')):
            
            # Input, output setting
            input_ids = input_[0].to(device, non_blocking=True)
            label = input_[-1].to(device, non_blocking=True)

            # Model
            model_out = model(input_ids)

            # Prediction
            prediction = torch.argmax(model_out, dim=1)

            # Decode
            sent = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            # Print loss value only training
            acc = sum(label.view(-1) == prediction.view(-1)) / len(label.view(-1))
            acc = acc.item() * 100
            test_acc += acc
            write_log(logger, 'Test Batch Accuracy: %3.2f%%' % acc)

            # Append
            total_sentence_list.append(sent)
            total_label_list.append(label.tolist())
            total_prediction_list.append(prediction.cpu().tolist())
    # Save
    data_name = f'{args.dataset}_{args.classifier_model_type}.csv'
    aug_dat = pd.DataFrame({
        'description': total_sentence_list,
        'label': total_label_list,
        'prediction': total_prediction_list
    }).to_csv(os.path.join(args.augmentation_path, data_name))

    test_acc /= len(dataloader_dict['test'])
    write_log(logger, 'Test Total Accuracy: %3.2f%%' % test_acc)