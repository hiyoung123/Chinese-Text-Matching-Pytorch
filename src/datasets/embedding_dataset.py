#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, dataset, vocab, max_seq_len, train=True):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.train = train
        self.dataset = self.process_data(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {key: value for key, value in self.dataset[idx].items()}

    def process_data(self, dataset):
        result = []
        for i, (seq1, seq2) in enumerate(zip(dataset['seq1'], dataset['seq2'])):
            ''' 分词 ？'''
            seq1 = [x for x in seq1.split()]
            seq2 = [x for x in seq2.split()]

            if len(seq1) > self.max_seq_len:
                seq1 = seq1[0:self.max_seq_len]
            if len(seq2) > self.max_seq_len:
                seq2 = seq2[0:self.max_seq_len]

            seq1 = [self.vocab[x] for x in seq1 if x in self.vocab]
            seq2 = [self.vocab[x] for x in seq2 if x in self.vocab]

            padding = [0] * (self.max_seq_len - len(seq1))
            seq1 += padding
            padding = [0] * (self.max_seq_len - len(seq2))
            seq2 += padding

            output = {
                'seq1': seq1,
                'seq2': seq2
            }
            if self.train:
                output.update({'label': dataset['label'].iloc[i]})
            result.append({key: torch.tensor(value) for key, value in output.items()})
        return result