# Import Modules
import os
import time
# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Import Custom Modules
from model.dataset import CustomDataset

def augmenting(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    print('Data Load & Setting!')
    with open(f'{args.preprocess_path}/{args.dataset}_{args.tokenizer}_preprocessed.pkl', 'rb') as f:
        data_ = pickle.load(f)
        train_input_ids = data_['train']['input_ids']
        train_token_type_ids = data_['train']['token_type_ids']
        train_attention_mask = data_['train']['attention_mask']
        train_comment_indices = data_['train_comment_indices']
        train_label = data_['train']['label']
        del data_

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(train_comment_indices, train_label, 
                               min_len=args.min_len, max_len=args.max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(), drop_last=False,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)
    }
    print(f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")