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
from torch.utils.data import DataLoader
# Import Custom Modules
from model.classification.dataset import CustomDataset, PadCollate
from model.classification.model import Classifier
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

    write_log(logger, "Testing Start")

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data Open
    processed_path = os.path.join(args.preprocess_path, 
                                    f'{args.dataset}_{args.cls_tokenizer}_valid_ratio_{args.valid_split_ratio}_preprocessed.pkl')
    with open(processed_path, 'rb') as f:
        data_ = pickle.load(f)
        test_input_ids = data_['test']['input_ids']
        test_attention_mask = data_['test']['attention_mask']
        test_label = data_['test']['label']
        if args.cls_tokenizer in ['T5', 'Bart']:
            test_token_type_ids = None
        else:
            test_token_type_ids = data_['test']['token_type_ids']
        del data_

    # 3) Dataloader setting
    test_dataset = CustomDataset(tokenizer=args.cls_tokenizer,
                                 input_ids_list=test_input_ids,
                                 label_list=test_label,
                                 attention_mask_list=test_attention_mask,
                                 token_type_ids_list=test_token_type_ids,
                                 min_len=args.min_len, max_len=args.max_len)
    test_dataloader = DataLoader(test_dataset, collate_fn=PadCollate(args.cls_tokenizer),
                                 drop_last=False, batch_size=args.batch_size, shuffle=False,
                                 pin_memory=True, num_workers=args.num_workers)

    print_text = f"Total number of testsets iterations - {len(test_dataset)}, {len(test_dataloader)}"
    write_log(logger, print_text)

    #===================================#
    #============Model Load=============#
    #===================================#

    # 1) Model initiating
    write_log(logger, "Instantiating models...")
    model = Classifier(model_type=args.cls_model_type, isPreTrain=args.cls_PLM_use,
                       num_class=len(set(test_label)), tokenizer_type=args.cls_tokenizer)

    # 2) Model load
    write_log(logger, "Loading models...")
    save_name = f'{args.dataset}_{args.cls_model_type}_aug_{args.train_with_augmentation}_cls_checkpoint.pth.tar'
    if args.train_only_augmentation:
        save_name = f'{args.dataset}_{args.cls_model_type}_only_aug_cls_checkpoint.pth.tar'
    checkpoint = torch.load(os.path.join(args.save_path, save_name), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    model = model.to(device)
    del checkpoint

    #===================================#
    #=========Model Test Start==========#
    #===================================#

    write_log(logger, 'Test start!')

    total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        for i, input_ in enumerate(tqdm(test_dataloader, 
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

            # Accuracy & Loss
            acc = sum(labels == out.max(dim=1)[1]) / len(labels)
            acc = acc.item() * 100
            total_acc += acc

            loss = F.cross_entropy(out, labels)
            total_loss += loss

    total_loss /= len(test_dataloader)
    total_acc /= len(test_dataloader)
    write_log(logger, 'Test Loss: %3.3f' % total_loss)
    write_log(logger, 'Test Accuracy: %3.2f%%' % total_acc)