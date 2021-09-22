import random
# Import PyTorch
import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, comment_list, label_list, min_len=4, max_len=512):
        data = []
        exception_count = 0
        for comment, label in zip(comment_list, label_list):
            if min_len <= len(comment) <= max_len:
                data.append((comment, label))
            else:
                exception_count += 1
        
        self.data = tuple(data)
        self.num_data = len(self.data)
        print(f'Exception by minimum and maximum length: {exception_count}')
        
    def __getitem__(self, index):
        comment, label = self.data[index]
        return comment, label
    
    def __len__(self):
        return self.num_data

class PadCollate:
    def __init__(self, pad_index=0, sep_index=3, dim=0):
        self.dim = dim
        self.pad_index = pad_index
        self.sep_index = sep_index

    def pad_collate(self, batch):
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.LongTensor(*pad_size).fill_(self.pad_index)], dim=dim)

        def pack_sentence(sentences, masking=False):
            sentences_len = max(map(lambda x: len(x), sentences))
            if masking:
                sentences = masking_sentences(sentences)
            sentences = [pad_tensor(torch.LongTensor(seq), sentences_len, self.dim) for seq in sentences]
            sentences = torch.cat(sentences)
            sentences = sentences.view(-1, sentences_len)
            return sentences
        
        comment, label = zip(*batch)
        return  pack_sentence(comment), torch.LongTensor(label)
        
    def __call__(self, batch):
        return self.pad_collate(batch)