#!usr/bin/env python
#-*- coding:utf-8 -*-

import argparse
import os

from src.train import run_train
from src.evaluation import run_eval
from src.inference import run_inference
from utils.config import Config


def main(args):
    config = Config.parse(args.model)
    for k, v in vars(args).items():
        config.update({k: v})

    model_path = os.path.join(config.model_path, config.model + '_' + config.task_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    result_path = os.path.join(config.result_path, config.model + '_' + config.task_name)
    if not os.path.exists(model_path):
        os.mkdir(result_path)

    config.model_path = model_path + '/best_model.pt'
    config.result_path = result_path + '/result.csv'

    if args.do_train:
        run_train(config)

    if args.do_eval:
        run_eval(config)

    if args.do_predict:
        run_inference(config)