import sentencepiece as spm
from transformers import BertTokenizer

from WAE_Augmnetation.task.utils import read_data

def preprocessing(args):
    """
    Build vocabulary using sentencepiece library from dataset csv file

    Args:
        dataset_path (str): path to dataset folder
        dataset (str): option for specific dataset to use 
        data_column_index (int): column index of data to extract from dataset file
        split_ratio (float): ratio between valid/test split
        vocab_size (int): size of vocabulary
        pad_id (int): id of padding token
        unk_id (int): id of unknown token
        bos_id (int): id of begin of sentence token
        eos_id (int): id of end of sentence token
        sentencepiece_model: sentencepiece model type
    """

    # Data Load
    train_dat, test_dat = read_data(args.dataset)

    # Data Split
    train_dat, valid_dat = train_valid_split(train_dat)

    # SentencePiece; spm
    if args.tokenizer == 'spm':

        with open(f'{args.preprocess_path}/{args.dataset}_text.txt', 'w') as f:
            for text in train_dat['total_text']:
                f.write(f'{text}\n')

        # SentencePiece Training
        spm.SentencePieceProcessor()
        spm.SentencePieceTrainer.Train(
            f'--input={save_path_train} --model_prefix=m_spm '
            f'--model_type={args.sentencepiece_model} --character_coverage=0.9995 --vocab_size={args.vocab_size} '
            f'--pad_id={args.pad_idx} --unk_id={args.unk_idx} --bos_id={args.bos_idx} --eos_id={args.eos_idx} '
            f'--split_by_whitespace=true --user_defined_symbols=[SEP]')

        # Vocabularay setting
        vocab_list = list()
        with open(f'{args.output_path}/m_spm.vocab') as f:
            for line in f:
                vocab_list.append(line[:-1].split('\t')[0])
        word2id_dict = {w: i for i, w in enumerate(vocab_list)}

        # SentencePiece model load
        sp_kr = spm.SentencePieceProcessor()
        sp_kr.Load(f"{args.output_path}/m_spm.model")

        # Encoding
        train_indices = tuple(
            [args.bos_idx] + sp_kr.encode(
                                korean, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
            [args.eos_idx] for korean in train_dat['total_text']
        )
        valid_indices = tuple(
            [args.bos_idx] + sp_kr.encode(korean, out_type=int) + [args.eos_idx] for korean in valid_dat['total_text']
        )
        test_indices = tuple(
            [args.bos_idx] + sp_kr.encode(korean, out_type=int) + [args.eos_idx] for korean in test_dat['total_text']
        )

        # Segment encoding
        

    # BERT Tokenizer; BERT
    if args.tokenizer == 'BERT':

        # Load pre-trained tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        # Tokenizing
        encoded_dict = tokenizer(
            train_dat.tolist(),
            max_length=args.max_len,
            padding='max_length'
        )