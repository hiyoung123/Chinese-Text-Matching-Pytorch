#!usr/bin/env python
#-*- coding:utf-8 -*-

import pickle
import random
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader


MODEL_CLASSES = {

}


class BaseTrainer:
    def __init__(self, config, model):
        pass

    def train(self, train, dev):
        pass

    def train_epoch(self, data, epoch):
        pass

    def train_step(self, batch):
        pass


class EpochTrainer:
    pass


class StepTrainer:
    pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_embedding(config, vocab):
    embedding_matrix = np.zeros((len(vocab) + 1, config.embed_dim))
    embeddings_index = pickle.load(open(config.embedding_path, 'rb'))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return torch.Tensor(embedding_matrix)


def run_train(config):

    train = pd.read_csv(config.train_path)
    dev = pd.read_csv(config.dev_path)

    set_seed(config.seed)

    bert_tokenizer, bert_config, bert_model, model, dataset = MODEL_CLASSES[config.model]
    if config.get('embedding_path', False):
        tokenizer = pickle.load(open(config.vocab_path, 'rb'))
        config['embedding'] = build_embedding(config, tokenizer)
        model = model(config)
    else:
        tokenizer = bert_tokenizer.from_pretrained(config.pre_trained_model + '/vocab.txt')
        bert_config = bert_config.from_pretrained(config.pre_trained_model + '/bert_config.json')
        bert = bert_model.from_pretrained(config.pre_trained_model + '/pytorch_model.bin', config=bert_config)
        model = model(bert=bert, config=config)

    train = dataset(train, tokenizer, config.max_seq_len, True)
    train = DataLoader(train, batch_size=config.batch_size)
    dev = dataset(dev, tokenizer, config.max_seq_len, True)
    dev = DataLoader(dev, batch_size=config.batch_size)

    if config.save_by_step:
        trainer = StepTrainer(config, model)
    else:
        trainer = EpochTrainer(config, model)
    trainer.train(train, dev)