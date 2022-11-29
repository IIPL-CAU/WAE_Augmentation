import logging
import sentencepiece as spm
from transformers import BertTokenizer, T5Tokenizer, BartTokenizer

from utils import TqdmLoggingHandler, write_log

def nmt_preprocessing(args, train_dat, valid_dat, test_dat):
    """
    Build vocabulary using sentencepiece library from dataset csv file

    Using args:
        tokenizer (str): Tokenizer select; SentencePiece(spm) or BertTokenizer(BERT)
        preprocess_path (str): pre-processed file saving path
        dataset (str): option for specific dataset to use 
        valid_split_ratio (float): ratio between train/valid split
        vocab_size (int): size of vocabulary
        pad_idx (int): id of padding token
        unk_idx (int): id of unknown token
        bos_idx (int): id of begin of sentence token
        eos_idx (int): id of end of sentence token
        sentencepiece_model (str): sentencepiece model type
    """

    # Logger setting
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, "Pre-processing Start")


    # Data save in dictionary
    encoded_dict = dict()
    encoded_dict['train'] = dict()
    encoded_dict['valid'] = dict()
    encoded_dict['test'] = dict()

    write_log(logger, "Tokenizing Start")

    # SentencePiece; spm
    if args.nmt_tokenizer == 'spm':

        with open(f'{args.preprocess_path}/{args.dataset}_text.txt', 'w') as f:
            for text in train_dat['total_text']:
                f.write(f'{text}\n')

        # SentencePiece Training
        # spm.SentencePieceProcessor()
        spm.SentencePieceTrainer.Train(
            f'--input={args.preprocess_path}/{args.dataset}_text.txt --model_prefix=m_spm_src '
            f'--model_type={args.sentencepiece_model} --character_coverage=0.9995 --vocab_size={args.src_vocab_size} '
            f'--pad_id={args.pad_idx} --unk_id={args.unk_idx} --bos_id={args.bos_idx} --eos_id={args.eos_idx} '
            f'--split_by_whitespace=true --user_defined_symbols=[SEP]')
        spm.SentencePieceTrainer.Train(
            f'--input={args.preprocess_path}/{args.dataset}_text.txt --model_prefix=m_spm_trg '
            f'--model_type={args.sentencepiece_model} --character_coverage=0.9995 --vocab_size={args.trg_vocab_size} '
            f'--pad_id={args.pad_idx} --unk_id={args.unk_idx} --bos_id={args.bos_idx} --eos_id={args.eos_idx} '
            f'--split_by_whitespace=true --user_defined_symbols=[SEP]')

        # Vocabularay setting
        src_vocab_list = list()
        with open(f'{args.output_path}/m_spm_src.vocab') as f:
            for line in f:
                src_vocab_list.append(line[:-1].split('\t')[0])
        src_word2id_dict = {w: i for i, w in enumerate(src_vocab_list)}

        trg_vocab_list = list()
        with open(f'{args.output_path}/m_spm_trg.vocab') as f:
            for line in f:
                trg_vocab_list.append(line[:-1].split('\t')[0])
        trg_word2id_dict = {w: i for i, w in enumerate(trg_vocab_list)}

        # SentencePiece model load
        spm_src_model = spm.SentencePieceProcessor()
        spm_src_model.Load(f"{args.output_path}/m_spm_src.model")

        spm_trg_model = spm.SentencePieceProcessor()
        spm_trg_model.Load(f"{args.output_path}/m_spm_trg.model")

        # Encoding
        for src, trg in zip(train_dat['src'], train_dat['trg']):
            src_encoded_text = [args.bos_idx] + spm_src_model.encoder(
                src, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
                    [args.eos_idx]
            trg_encoded_text = [args.bos_idx] + spm_src_model.encoder(
                trg, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
                    [args.eos_idx]
            src_encoded_text = src_encoded_text + [args.pad_idx for _ in range(args.max_len - len(src_encoded_text))]
            trg_encoded_text = src_encoded_text + [args.pad_idx for _ in range(args.max_len - len(trg_encoded_text))]
            encoded_dict['train']['src_input_ids'].append(src_encoded_text)
            encoded_dict['train']['trg_input_ids'].append(src_encoded_text)
        for src, trg in zip(valid_dat['src'], valid_dat['trg']):
            src_encoded_text = [args.bos_idx] + spm_src_model.encoder(
                text, out_type=int) + [args.eos_idx]
            encoded_text = [args.bos_idx] + spm_trg_model.encoder(
                text, out_type=int) + [args.eos_idx]
            encoded_text = encoded_text + [args.pad_idx for _ in range(args.max_len - len(encoded_text))]
            encoded_text = encoded_text + [args.pad_idx for _ in range(args.max_len - len(encoded_text))]
            encoded_dict['valid']['input_ids'].append(encoded_text)
            encoded_dict['valid']['input_ids'].append(encoded_text)
        for text in test_dat['src']:
            encoded_text = [args.bos_idx] + spm_src_model.encoder(
                text, out_type=int) + [args.eos_idx]
            encoded_text = [args.bos_idx] + spm_src_model.encoder(
                text, out_type=int) + [args.eos_idx]
            encoded_text = encoded_text + [args.pad_idx for _ in range(args.max_len - len(encoded_text))]
            encoded_text = encoded_text + [args.pad_idx for _ in range(args.max_len - len(encoded_text))]
            encoded_dict['test']['input_ids'].append(encoded_text)
            encoded_dict['test']['input_ids'].append(encoded_text)

        # Segment encoding
        encoded_dict['train']['token_type_ids'] = list()
        encoded_dict['valid']['token_type_ids'] = list()
        encoded_dict['test']['token_type_ids'] = list()

        for ind in encoded_dict['train']['input_ids']:
            token_type_ids_ = [0 if i <= ind.index(4) else 1 for i in range(len(ind))]
            token_type_ids_ = token_type_ids_ + [0 for _ in range(args.max_len - len(ind))]
            encoded_dict['train']['token_type_ids'].append(token_type_ids_)
        for ind in encoded_dict['valid']['input_ids']:
            token_type_ids_ = [0 if i <= ind.index(4) else 1 for i in range(len(ind))]
            token_type_ids_ = token_type_ids_ + [0 for _ in range(args.max_len - len(ind))]
            encoded_dict['valid']['token_type_ids'].append(token_type_ids_)
        for ind in encoded_dict['test']['input_ids']:
            token_type_ids_ = [0 if i <= ind.index(4) else 1 for i in range(len(ind))]
            token_type_ids_ = token_type_ids_ + [0 for _ in range(args.max_len - len(ind))]
            encoded_dict['test']['token_type_ids'].append(token_type_ids_)

        # Attention mask encoding
        encoded_dict['train']['attention_mask'] = list()
        encoded_dict['valid']['attention_mask'] = list()
        encoded_dict['test']['attention_mask'] = list()

        for ind in encoded_dict['train']['input_ids']:
            encoded_dict['train']['attention_mask'].append([1 if i <= ind.index(args.eos_idx) else 0 for i in range(args.max_len)])
        for ind in encoded_dict['valid']['input_ids']:
            encoded_dict['valid']['attention_mask'].append([1 if i <= ind.index(args.eos_idx) else 0 for i in range(args.max_len)])
        for ind in encoded_dict['test']['input_ids']:
            encoded_dict['test']['attention_mask'].append([1 if i <= ind.index(args.eos_idx) else 0 for i in range(args.max_len)])

        # Label setting
        encoded_dict['train']['label'] = train_dat['label']
        encoded_dict['valid']['label'] = valid_dat['label']
        encoded_dict['test']['label'] = test_dat['label']

    # Huggingface
    else:

        # Load pre-trained tokenizer
        if args.nmt_tokenizer == 'BERT':
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        elif args.nmt_tokenizer == 'T5':
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
            tokenizer.add_tokens('[SEP]')
        elif args.nmt_tokenizer == 'Bart':
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
            tokenizer.add_tokens('[SEP]')

        # Tokenizing
        if len(train_dat.columns) > 3:
            if args.nmt_tokenizer == 'BERT':
                # Train data
                encoded_dict['train'] = tokenizer(
                    train_dat['title'].tolist(),
                    train_dat['description'].tolist(),
                    max_length=args.max_len,
                    padding='max_length',
                    truncation=True
                )

                # Validation data
                encoded_dict['valid'] = tokenizer(
                    valid_dat['title'].tolist(),
                    valid_dat['description'].tolist(),
                    max_length=args.max_len,
                    padding='max_length',
                    truncation=True
                )
                # Test data
                encoded_dict['test'] = tokenizer(
                    test_dat['title'].tolist(),
                    test_dat['description'].tolist(),
                    max_length=args.max_len,
                    padding='max_length',
                    truncation=True
                )
            else:
                # Train data
                encoded_dict['train'] = tokenizer(
                    train_dat['total_text'].tolist(),
                    max_length=args.max_len,
                    padding='max_length',
                    truncation=True
                )

                # Validation data
                encoded_dict['valid'] = tokenizer(
                    valid_dat['total_text'].tolist(),
                    max_length=args.max_len,
                    padding='max_length',
                    truncation=True
                )

                # Test data
                encoded_dict['test'] = tokenizer(
                    test_dat['total_text'].tolist(),
                    max_length=args.max_len,
                    padding='max_length',
                    truncation=True
                )

        else:
            # Train data
            encoded_dict['train'] = tokenizer(
                train_dat['description'].tolist(),
                max_length=args.max_len,
                padding='max_length',
                truncation=True
            )

            # Validation data
            encoded_dict['valid'] = tokenizer(
                valid_dat['description'].tolist(),
                max_length=args.max_len,
                padding='max_length',
                truncation=True
            )

            # Test data
            encoded_dict['test'] = tokenizer(
                test_dat['description'].tolist(),
                max_length=args.max_len,
                padding='max_length',
                truncation=True
            )

        # Label setting
        encoded_dict['train']['label'] = train_dat['label']
        encoded_dict['valid']['label'] = valid_dat['label']
        encoded_dict['test']['label'] = test_dat['label']

    # Saving
    write_log(logger, "Saving Start")