import argparse
from time import time

# Import custom modules
from task.preprocessing import preprocessing
from task.classification.augment_training import augment_training
from task.classification.augmentation import augmentation
from task.classification.training import training
from task.classification.testing import testing
from utils import str2bool, path_check

def main(args):

    # Path setting
    path_check(args)

    # Time setting
    total_start_time = time()

    # Task run
    if args.task == 'CLS':
        if args.preprocessing:
            preprocessing(args)

        if args.augment_training:
            augment_training(args)

        if args.augmentation:
            augmentation(args)

        if args.training:
            training(args)

        if args.testing:
            testing(args)

    if args.task == 'NMT':
        if args.preprocessing:
            preprocessing(args)

    # Time calculate
    print(f'Done! ; {round((time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    # Task setting
    parser.add_argument('--task', default='NMT', type=str, choices=['NMT', 'CLS'])
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--augment_training', action='store_true')
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Data setting
    data_ = ['IMDB', 'Yelp_Full', 'DBpedia', 'AG_News', 'SST2', 'SST5', 'ProsCons', 'SUBJ', 'TREC', 'MR']
    parser.add_argument('--dataset', type=str, choices=data_,
                        help='Dataset select; [IMDB, Yelp_Full, DBpedia, AG_News, SST2, SST5, ProsCons, SUBJ, TREC, MR]')
    # Path setting
    parser.add_argument('--data_path', default='/HDD/dataset/text_classification', type=str,
                        help='Original data path')
    parser.add_argument('--preprocess_path', default='/HDD/kyohoon/WAE/preprocessing', type=str,
                        help='Preprocessed data  file path')
    parser.add_argument('--save_path', default='/HDD/kyohoon/model_checkpoint/WAE/', type=str,
                        help='Model checkpoint file path')
    parser.add_argument('--augmentation_path', default='/HDD/kyohoon/WAE/augmentation', type=str,
                        help='Augmented file path')
    # Preprocessing setting
    parser.add_argument('--sentencepiece_model', default='unigram', choices=['unigram', 'bpe', 'word', 'char'],
                        help="Google's SentencePiece model type; Default is unigram")
    parser.add_argument('--valid_split_ratio', default=0.05, type=float,
                        help='Validation dataset split ratio; Default is 0.05')
    parser.add_argument('--vocab_size', default=8000, type=int, 
                        help='Source language vocabulary size; Default is 8000')
    parser.add_argument('--pad_idx', default=0, type=int,
                        help='Padding token index; Default is 0')
    parser.add_argument('--unk_idx', default=3, type=int,
                        help='Unknown token index; Default is 3')
    parser.add_argument('--bos_idx', default=1, type=int,
                        help='Padding token index; Default is 1')
    parser.add_argument('--eos_idx', default=2, type=int,
                        help='Padding token index; Default is 2')
    parser.add_argument('--min_len', default=4, type=int,
                        help='Minimum length of sequence; Default is 4')
    parser.add_argument('--max_len', default=300, type=int,
                        help='Maximum length of sequence; Default is 300')
    # WAE setting
    parser.add_argument('--ae_type', default='WAE', type=str, choices=['WAE', 'VAE'],
                        help='Auto-encoder type; Default is WAE')
    parser.add_argument('--aug_tokenizer', default='T5', type=str, choices=['BERT', 'T5', 'spm', 'Bart'],
                        help='Tokenizer settings; Default is T5')
    parser.add_argument('--aug_model_type', default='T5', type=str, choices=['BERT','T5', 'Bart','BERT+T5'],
                        help='Model settings; Default is T5')
    parser.add_argument('--aug_PLM_use', default=False, type=str2bool,
                        help='Model settings; Default is T5')
    parser.add_argument('--WAE_decoder', default='Transformer', type=str, choices=['Transformer', 'LSTM', 'GRU'],
                        help='Decoder Type; Default is Transformer')
    parser.add_argument('--WAE_loss', default='mmd', choices=['mmd', 'gan'],
                        help='Wasserstein Auto-encoder Loss Type; Default is mmd')
    parser.add_argument('--d_latent', default=256, type=int,
                        help='Latent space dimension; Default is 256')
    parser.add_argument('--z_var', default=2, type=int,
                        help='Default is 2')
    parser.add_argument('--loss_lambda', default=1000, type=int,
                        help='MMD loss lambda; Default is 1000')
    # VAE setting
    parser.add_argument('--vae_beta', default=10, type=int,
                        help='Default is 10')
    # Training setting
    parser.add_argument('--train_with_augmentation', default=True, type=str2bool,
                        help='Text classifier training with augmentation data; Default is True')
    parser.add_argument('--train_only_augmentation', default=False, type=str2bool,
                        help='Text classifier training only with augmentation data; Default is False')
    parser.add_argument('--cls_tokenizer', default='BERT', type=str, choices=['BERT', 'T5', 'spm', 'Bart'],
                        help='Tokenizer settings; Default is T5')
    parser.add_argument('--cls_model_type', default='BERT', type=str, choices=['CNN', 'RNN', 'BERT'],
                        help='Classifier model settings; Default is CNN')
    parser.add_argument('--cls_PLM_use', default=True, type=str2bool,
                        help='Model settings; Default is T5')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Num CPU Workers; Default is 8')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size; Default is 16')
    parser.add_argument('--num_epochs', default=100, type=int, 
                        help='Epoch count; Default is 100')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='Maximum learning rate of warmup scheduler; Default is 5e-5')
    parser.add_argument('--w_decay', default=1e-5, type=float,
                        help="Ralamb's weight decay; Default is 1e-5")
    parser.add_argument('--label_smoothing', default=0.05, type=float,
                        help="Label smoothing ratio; Default is 0.05")
    parser.add_argument('--clip_grad_norm', default=5, type=int, 
                        help='Graddient clipping norm; Default is 5')
    # Optimizer & LR_Scheduler setting
    optim_list = ['AdamW', 'Adam', 'SGD', 'Ralamb']
    scheduler_list = ['constant', 'warmup', 'reduce_train', 'reduce_valid', 'lambda']
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=optim_list,
                        help="Choose optimizer setting in 'AdamW', 'Adam', 'SGD', 'Ralamb'; Default is AdamW")
    parser.add_argument('--scheduler', default='constant', type=str, choices=scheduler_list,
                        help="Choose optimizer setting in 'constant', 'warmup', 'reduce'; Default is constant")
    parser.add_argument('--n_warmup_epochs', default=2, type=int, 
                        help='Wamrup epochs when using warmup scheduler; Default is 2')
    parser.add_argument('--lr_lambda', default=0.95, type=float,
                        help="Lambda learning scheduler's lambda; Default is 0.95")
    # Print frequency
    parser.add_argument('--print_freq', default=100, type=int, 
                        help='Print training process frequency; Default is 100')
    args = parser.parse_args()

    main(args)