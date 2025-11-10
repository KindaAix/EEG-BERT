import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F

import argparse

from transformers import HubertForSequenceClassification, TrainingArguments
from models.bert_arch import BERT_arch as bert
from uilts.load_data import get_train_loader, get_valid_loader
from models.loss import EEGLoss


def get_config():
    parser = argparse.ArgumentParser(description="EEG EMO BERT")
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--data_root', type=str, default=None, help='数据目录')
    parser.add_argument('--save_dir', type=str, default=None, help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument("--kmeans_path", type=str, default=None, help='Torch-KMeans权重文件')

    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = json.load(f)    
    else:
        cfg = {}

    config = {}
    config['data_root']    = args.data_root   or cfg.get('data_root', 'dataset/')
    config['save_dir']    = args.save_dir   or cfg.get('save_dir', 'checkpoints/')
    config['epochs']      = args.epochs     or cfg.get('epochs', 50)
    config['batch_size']  = args.batch_size or cfg.get('batch_size', 32)
    config['lr']          = args.lr         or cfg.get('lr', 1e-3)
    config['kmeans_path'] = args.kmeans     or cfg.get('kmeans_path', 'KMeans.pt')
    cfg.update(config)

    return cfg


def main():
    cfg = get_config()

    train_loader = get_train_loader(data_root=cfg["data_root"], kmeans_path=cfg["kmeans_path"],
                                    batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    val_loader = get_valid_loader(data_root=cfg["data_root"], kmeans_path=cfg["kmeans_path"],
                                  batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])


    model = bert()
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_steps=50,
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=val_loader,
        compute_loss_func=EEGLoss().loss(),
        compute_metrics=None,
        optimizers=(torch.c, None),
    )

