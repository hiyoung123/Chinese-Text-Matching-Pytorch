#!usr/bin/env python
#-*- coding:utf-8 -*-

import pickle
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader


MODEL_CLASSES = {

}


class Evaluator:
    def __init__(self, config, model):
        pass

    def evaluate(self, data):
        pass


def set_seed(seed):
    # seed = 7874
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


def run_eval(config):

    test = pd.read_csv(config.test_path)

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

    test = dataset(test, tokenizer, config.max_seq_len, True)
    test = DataLoader(test, batch_size=config.batch_size)
    evaluator = Evaluator(config, model)
    result = evaluator.evaluate(test)
    # print(acc)