from task.preprocessing.utils import read_data
from task.preprocessing.nmt_preprocessing import nmt_preprocessing
from task.preprocessing.cls_preprocessing import cls_preprocessing

def preprocessing(args):

    # Data Load
    train_dat, valid_dat, test_dat, label_dict = read_data(args.dataset, args.data_path)
    
    if args.task == 'CLS':
        cls_preprocessing(args, train_dat, test_dat, label_dict)

    if args.task == 'NMT':
        nmt_preprocessing(args, train_dat, valid_dat, test_dat)